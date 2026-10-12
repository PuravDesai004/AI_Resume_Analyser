"""Gemini -> Groq generation fallback adapted from the separate RAG project."""
import logging
import os
from dotenv import load_dotenv
load_dotenv(override=True)

from google import genai
from google.genai import types
from chat_schemas import GeneratedChatAnswer
from rag_engine.generation import CHAT_SYSTEM_PROMPT, build_prompt

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        self._client = None
        self._groq_client = None
        self._gemini_disabled = False

    @property
    def client(self):
        if self._client is None and not self._gemini_disabled:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                self._gemini_disabled = True
                return None
            try:
                self._client = genai.Client(
                    api_key=api_key,
                    http_options={"timeout": 15000, "retry_options": {"attempts": 1}},
                )
            except Exception:
                self._gemini_disabled = True
        return self._client

    def run(self, query_text, retrieved_chunks, history=()):
        prompt = build_prompt(query_text, retrieved_chunks, history)
        if not self._gemini_disabled and self.client:
            try:
                response = self.client.models.generate_content(
                    model=os.getenv("RAG_GENERATION_MODEL", "gemini-3.5-flash-lite"),
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=CHAT_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_json_schema=GeneratedChatAnswer.model_json_schema(),
                        temperature=0.2,
                    ),
                )
                return GeneratedChatAnswer.model_validate_json(response.text or "")
            except Exception as exc:
                logger.warning(
                    "Gemini chat generation failed (%s, HTTP %s); checking configured fallback.",
                    type(exc).__name__, getattr(exc, "code", "n/a"),
                )
                code = getattr(exc, "code", None)
                if code in (400, 401, 403) or "unauthenticated" in str(exc).lower():
                    self._gemini_disabled = True
        if os.getenv("GROQ_API_KEY"):
            try:
                from groq import Groq
                if self._groq_client is None:
                    self._groq_client = Groq(
                        api_key=os.getenv("GROQ_API_KEY"), timeout=30.0, max_retries=0,
                    )
                response = self._groq_client.chat.completions.create(
                    model=os.getenv("RAG_GROQ_MODEL", "openai/gpt-oss-20b"),
                    messages=[
                        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"}, temperature=0.2,
                )
                return GeneratedChatAnswer.model_validate_json(
                    response.choices[0].message.content or "",
                )
            except Exception:
                logger.warning("Groq chat generation failed.")
        raise RuntimeError("Chat models are unavailable or returned an invalid answer. Please retry.")
