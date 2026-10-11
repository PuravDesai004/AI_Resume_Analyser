"""Bounded contracts for resume/JD chat."""
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, model_validator
from schemas import DetailAnalysis


class ChatPage(BaseModel):
    text: str = Field(min_length=1, max_length=120_000)
    page: int | None = Field(default=None, ge=1)


class ChatDocument(BaseModel):
    document_id: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=250)
    pages: list[ChatPage] = Field(min_length=1, max_length=200)
    backend_id: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def readable_text(self):
        if not any(p.text.strip() for p in self.pages):
            raise ValueError("Document has no readable text. A scanned PDF may need OCR.")
        if sum(len(p.text) for p in self.pages) > 120_000:
            raise ValueError("Document text exceeds the chat limit of 120,000 characters.")
        return self


class ChatContextRequest(BaseModel):
    session_id: UUID
    resume: ChatDocument
    jd: ChatDocument | None = None
    analysis: DetailAnalysis | None = None

    @model_validator(mode="after")
    def analysis_matches_documents(self):
        if self.analysis:
            if self.analysis.resume_id != self.resume.document_id:
                raise ValueError("Analysis belongs to a different resume.")
            if not self.jd or self.jd.backend_id != self.analysis.jd_id:
                raise ValueError("Analysis belongs to a different job description.")
            if len(self.analysis.model_dump_json()) > 16_000:
                raise ValueError("Analysis exceeds the chat context limit.")
        return self


class ChatContextResponse(BaseModel):
    context_id: str
    chunk_count: int
    reused: bool


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4_000)


class ChatRequest(BaseModel):
    session_id: UUID
    context_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    message: str = Field(min_length=1, max_length=2_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def nonempty_message(self):
        self.message = self.message.strip()
        if not self.message:
            raise ValueError("Please enter a question.")
        return self


class GeneratedChatAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1, max_length=8_000)
    context_sufficient: bool
    cited_chunk_ids: list[str] = Field(default_factory=list, max_length=20)


class ChatCitation(BaseModel):
    citation_id: int
    chunk_id: str
    source: str
    document_type: Literal["resume", "jd", "analysis"]
    page: int | None = None
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    context_sufficient: bool
    citations: list[ChatCitation] = Field(default_factory=list)
