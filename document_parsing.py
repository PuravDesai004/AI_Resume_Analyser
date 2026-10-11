"""Extract document text and real PDF page references for analysis and chat."""
import io
from pathlib import Path
import docx
import pymupdf


def extract_document(file_bytes: bytes, filename: str):
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
            pages = [
                {"text": page.get_text().strip(), "page": index}
                for index, page in enumerate(pdf, 1)
                if page.get_text().strip()
            ]
    elif extension == ".docx":
        document = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join(
            [p.text for p in document.paragraphs]
            + [" | ".join(cell.text for cell in row.cells)
               for table in document.tables for row in table.rows]
        ).strip()
        pages = [{"text": text, "page": None}] if text else []
    elif extension == ".txt":
        text = file_bytes.decode("utf-8").strip()
        pages = [{"text": text, "page": None}] if text else []
    else:
        raise ValueError("Please use PDF, DOCX, or UTF-8 TXT files.")
    return "\n".join(p["text"] for p in pages).strip(), pages
