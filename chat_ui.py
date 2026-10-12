"""Streamlit connection to the session-scoped document RAG API."""
import hashlib
import json
from uuid import uuid4
import requests
import streamlit as st


def init_chat_state():
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid4())
    for name in ("chat_contexts", "chat_messages"):
        if name not in st.session_state:
            st.session_state[name] = {}


def error_detail(error):
    if error.response is not None:
        try:
            detail = error.response.json().get("detail")
            if isinstance(detail, str):
                return detail
            if isinstance(detail, list) and detail:
                return detail[0].get("msg", "The selected documents could not be processed.")
        except ValueError:
            pass
    return "Chat could not reach the document service. Check the backend and retry."


def reset_chat(api_url):
    init_chat_state()
    if st.session_state.chat_contexts:
        try:
            response = requests.delete(
                f"{api_url}/chat/sessions/{st.session_state.chat_session_id}", timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            st.error(error_detail(exc))
            return False
    st.session_state.chat_contexts = {}
    st.session_state.chat_messages = {}
    st.session_state.chat_session_id = str(uuid4())
    return True


def remove_chat_document(api_url, document_id):
    init_chat_state()
    if not st.session_state.chat_contexts:
        return True
    try:
        response = requests.delete(
            f"{api_url}/chat/documents/{document_id}",
            params={"session_id": st.session_state.chat_session_id}, timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(error_detail(exc))
        return False
    for key, entry in list(st.session_state.chat_contexts.items()):
        if document_id in entry["document_ids"]:
            st.session_state.chat_contexts.pop(key)
            st.session_state.chat_messages.pop(key, None)
    return True


def document_payload(record, source):
    return {
        "document_id": record["id"], "source": source,
        "pages": record.get("pages") or [{"text": record["text"], "page": None}],
        "backend_id": record.get("backend_id"),
    }


def render_answer(message):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        citations = message.get("citations", [])
        if citations:
            with st.expander("Sources"):
                for citation in citations:
                    page = f" · page {citation['page']}" if citation.get("page") else ""
                    kind = {"resume": "Resume", "jd": "Job description", "analysis": "Analysis"}
                    st.caption(
                        f"[{citation['citation_id']}] {kind[citation['document_type']]}: "
                        f"{citation['source']}{page}"
                    )
                    st.write(citation["excerpt"])


def render_chat(api_url, resumes, jds, get_analysis):
    init_chat_state()
    st.header("Chat about your resume and job application")
    st.caption("Ask about the selected documents, compare requirements, or explain your analysis.")
    if not resumes:
        st.info("Upload a resume in the first tab to start chatting.")
        return
    if st.session_state.get("chat_resume_selection", 0) not in range(len(resumes)):
        st.session_state.chat_resume_selection = 0
    if st.session_state.get("chat_jd_selection", -1) not in [-1] + list(range(len(jds))):
        st.session_state.chat_jd_selection = -1
    resume_index = st.selectbox(
        "Resume for this conversation", range(len(resumes)),
        format_func=lambda i: resumes[i]["name"], key="chat_resume_selection",
    )
    job_index = st.selectbox(
        "Job description (optional)", [-1] + list(range(len(jds))),
        format_func=lambda i: "Resume only" if i == -1 else f"{jds[i]['title']} @ {jds[i]['company']}",
        key="chat_jd_selection",
    )
    resume = resumes[resume_index]
    jd = jds[job_index] if job_index >= 0 else None
    analysis = get_analysis(resume["id"], jd["backend_id"]) if jd and jd.get("backend_id") else None
    if analysis and ("error" in analysis or "error_code" in analysis):
        analysis = None
    if not resume["text"].strip() or (jd and not jd["text"].strip()):
        st.warning("The selected document has no readable text. Scanned PDFs need OCR.")
        return
    if jd:
        st.caption("Placement analysis included." if analysis else "Run gap analysis to include its scores and findings.")
    payload = {
        "session_id": st.session_state.chat_session_id,
        "resume": document_payload(resume, resume["name"]),
        "jd": document_payload(jd, jd.get("source", jd["title"])) if jd else None,
        "analysis": analysis,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    messages = st.session_state.chat_messages.setdefault(key, [])
    if st.button("Clear this conversation", key="chat_clear"):
        st.session_state.chat_messages[key] = []
        st.rerun()
    for message in messages:
        render_answer(message)
    question = st.chat_input("Ask a question about your selected resume or role", max_chars=2000)
    if not question:
        return
    render_answer({"role": "user", "content": question})
    with st.spinner("Looking through your selected documents..."):
        try:
            if key not in st.session_state.chat_contexts:
                response = requests.post(f"{api_url}/chat/documents", json=payload, timeout=(5, 180))
                response.raise_for_status()
                st.session_state.chat_contexts[key] = {
                    "context_id": response.json()["context_id"],
                    "document_ids": [resume["id"]] + ([jd["id"]] if jd else []),
                }
            history = [{"role": m["role"], "content": m["content"][:4000]} for m in messages[-6:]]
            response = requests.post(f"{api_url}/chat", json={
                "session_id": st.session_state.chat_session_id,
                "context_id": st.session_state.chat_contexts[key]["context_id"],
                "message": question, "history": history,
            }, timeout=(5, 120))
            if response.status_code == 404:
                st.session_state.chat_contexts.pop(key, None)
            response.raise_for_status()
            answer = response.json()
        except requests.Timeout:
            st.warning("Chat took too long. Please retry; successful indexing will be reused.")
            return
        except requests.RequestException as exc:
            st.error(error_detail(exc))
            return
    assistant = {"role": "assistant", "content": answer["answer"], "citations": answer["citations"]}
    messages.extend([{"role": "user", "content": question}, assistant])
    render_answer(assistant)
