"""PDF processing pipeline: extract text from PDFs with metadata preservation."""

import re
from typing import Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings


class PDFProcessor:
    """Processes PDF files into text chunks with metadata (page numbers, sections)."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Settings.CHUNK_SIZE,
            chunk_overlap=Settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def extract_text_with_metadata(self, pdf_path: str) -> list[dict]:
        """Extract text from PDF, preserving page numbers and section headings.

        Returns a list of dicts with keys: text, page_number, section_title, source.
        """
        doc = fitz.open(pdf_path)
        pages_data = []

        current_section = "Introduction"

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            page_text = ""
            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    line_text = ""
                    max_font_size = 0
                    is_bold = False

                    for span in line["spans"]:
                        line_text += span["text"]
                        max_font_size = max(max_font_size, span["size"])
                        if "bold" in span["font"].lower() or "Bold" in span["font"]:
                            is_bold = True

                    line_text = line_text.strip()
                    if not line_text:
                        continue

                    # Detect section headings: bold text or large font size
                    if (is_bold or max_font_size > 13) and len(line_text) < 200:
                        heading = self._clean_heading(line_text)
                        if heading:
                            current_section = heading

                    page_text += line_text + "\n"

            # Clean the page text
            page_text = self._clean_text(page_text)

            if page_text.strip():
                pages_data.append(
                    {
                        "text": page_text,
                        "page_number": page_num + 1,  # 1-indexed
                        "section_title": current_section,
                        "source": pdf_path.split("/")[-1],
                    }
                )

        doc.close()
        return pages_data

    def process_pdf(self, pdf_path: str) -> list[dict]:
        """Full pipeline: extract text from PDF, chunk it, and return chunks with metadata."""
        pages_data = self.extract_text_with_metadata(pdf_path)

        all_chunks = []
        chunk_id = 0

        for page_data in pages_data:
            text = page_data["text"]
            if not text.strip():
                continue

            chunks = self.text_splitter.split_text(text)

            for chunk_text in chunks:
                if not chunk_text.strip():
                    continue

                # Try to detect a more specific section within the chunk
                section = self._detect_section_in_chunk(
                    chunk_text, page_data["section_title"]
                )

                all_chunks.append(
                    {
                        "text": chunk_text,
                        "page_number": page_data["page_number"],
                        "section_title": section,
                        "source": page_data["source"],
                        "chunk_id": f"chunk_{chunk_id}",
                    }
                )
                chunk_id += 1

        return all_chunks

    def _clean_text(self, text: str) -> str:
        """Remove headers, footers, standalone page numbers, and excessive whitespace."""
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            # Skip standalone page numbers
            if re.match(r"^\d{1,4}$", stripped):
                continue
            # Skip common header/footer patterns
            if re.match(r"^(Page \d+|Chapter \d+|\d+\s*\|)", stripped, re.IGNORECASE):
                continue
            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _clean_heading(self, text: str) -> Optional[str]:
        """Clean and validate a potential section heading."""
        text = text.strip()
        # Remove numbering prefix like "1.1" or "Chapter 3:"
        text = re.sub(r"^[\d.]+\s*", "", text)
        text = re.sub(r"^Chapter\s+\d+[:\s]*", "", text, flags=re.IGNORECASE)
        text = text.strip()

        if len(text) < 3 or len(text) > 150:
            return None
        return text

    def _detect_section_in_chunk(self, chunk_text: str, default_section: str) -> str:
        """Try to find a section heading within the chunk text."""
        lines = chunk_text.split("\n")
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            if line and line.isupper() and len(line) < 100:
                return line
        return default_section
