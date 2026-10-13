"""Validate upload batches without losing errors when the dashboard reruns."""
from pathlib import Path
from uuid import uuid4

from document_parsing import extract_document


def prepare_uploads(files, existing, kind, validate_text, max_file_size, cap=10):
    if kind not in ("resume", "jd"):
        raise ValueError("Unknown document type.")
    records, notices = [], []
    # PDF extraction and the fixture loader differ in surrounding whitespace.
    known_texts = {" ".join(record["text"].split()) for record in existing}
    for uploaded in files:
        try:
            content = uploaded.getvalue()
            if len(content) > max_file_size:
                notices.append(("error", f"'{uploaded.name}' exceeds the 5MB size limit. Upload rejected."))
                continue
            text, pages = extract_document(content, uploaded.name)
            if not text:
                notices.append(("error", f"'{uploaded.name}' contains no readable text. Scanned PDFs need OCR."))
                continue
            normalized_text = " ".join(text.split())
            if normalized_text in known_texts:
                notices.append(("info", f"'{uploaded.name}' is already loaded. Duplicate skipped."))
                continue
            if cap is not None and len(existing) + len(records) >= cap:
                notices.append(("warning", f"Cannot add '{uploaded.name}': the {kind} limit of {cap} is reached. Remove a document first."))
                continue
            status, warning = validate_text(text)
            record = {
                "id": f"{'res' if kind == 'resume' else 'jd'}_{uuid4().hex}",
                "text": text, "pages": pages, "words": len(text.split()),
                "status": status, "warning": warning,
            }
            if kind == "resume":
                record["name"] = uploaded.name
            else:
                record.update(
                    title=Path(uploaded.name).stem.title(), source=uploaded.name,
                    company="Partner Company", location="Remote / Hybrid", backend_id=None,
                )
            records.append(record)
            known_texts.add(normalized_text)
            notices.append(("success", f"Added '{uploaded.name}'."))
        except Exception as exc:
            notices.append(("error", f"Could not read '{uploaded.name}': {exc}"))
    return records, notices
