"""PDF file loader using PyMuPDF."""

import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from PIL import Image

from .base import BaseLoader, LoadedDocument

logger = logging.getLogger(__name__)


@dataclass
class PageImage:
    """A rendered page image from a PDF."""

    page_num: int
    image: Image.Image
    width: int
    height: int
    has_text: bool = True
    has_images: bool = False


@dataclass
class PDFWithImages(LoadedDocument):
    """PDF document with extracted page images for ColPali."""

    page_images: list[PageImage] = field(default_factory=list)


class PDFLoader(BaseLoader):
    """Loader for PDF files using PyMuPDF (fitz)."""

    SUPPORTED_EXTENSIONS = {".pdf"}
    DEFAULT_DPI = 144  # Good balance between quality and size

    def load(self, source: str | Path | BinaryIO) -> LoadedDocument:
        """Load a PDF file.

        Args:
            source: Path to PDF file or file-like object

        Returns:
            LoadedDocument with extracted text content
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF loading. "
                "Install with: pip install pymupdf"
            )

        # Handle different source types
        if isinstance(source, (str, Path)):
            path = Path(source)
            doc = fitz.open(path)
            source_name = str(path)
            title = self._get_title_from_path(path)
        else:
            # File-like object
            if hasattr(source, "read"):
                data = source.read()
                if isinstance(data, str):
                    data = data.encode("utf-8")
                doc = fitz.open(stream=data, filetype="pdf")
            else:
                doc = fitz.open(stream=source, filetype="pdf")
            source_name = getattr(source, "name", "uploaded_file.pdf")
            title = Path(source_name).stem

        # Extract text from all pages
        pages_text = []
        page_count = len(doc)

        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages_text.append(text)

        doc.close()

        content = "\n\n".join(pages_text)

        # Try to get title from PDF metadata
        pdf_title = doc.metadata.get("title") if hasattr(doc, "metadata") else None
        if pdf_title:
            title = pdf_title

        metadata = {
            "format": "pdf",
            "page_count": page_count,
            "pages_with_text": len(pages_text),
        }

        logger.info(
            f"Loaded PDF: {source_name} ({page_count} pages, {len(content)} chars)"
        )

        return LoadedDocument(
            content=content,
            source=source_name,
            title=title,
            metadata=metadata,
        )

    def supports(self, source: str | Path) -> bool:
        """Check if source is a PDF file."""
        path = Path(source)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load_with_images(
        self,
        source: str | Path | BinaryIO,
        dpi: int | None = None,
        max_pages: int | None = None,
    ) -> PDFWithImages:
        """Load a PDF file with page images for ColPali.

        Args:
            source: Path to PDF file or file-like object
            dpi: Resolution for rendering (default 144)
            max_pages: Maximum pages to render (None = all)

        Returns:
            PDFWithImages with text and page images
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF loading. "
                "Install with: pip install pymupdf"
            )

        dpi = dpi or self.DEFAULT_DPI
        zoom = dpi / 72  # PyMuPDF uses 72 DPI as base

        # Handle different source types
        if isinstance(source, (str, Path)):
            path = Path(source)
            doc = fitz.open(path)
            source_name = str(path)
            title = self._get_title_from_path(path)
        else:
            if hasattr(source, "read"):
                data = source.read()
                if isinstance(data, str):
                    data = data.encode("utf-8")
                doc = fitz.open(stream=data, filetype="pdf")
            else:
                doc = fitz.open(stream=source, filetype="pdf")
            source_name = getattr(source, "name", "uploaded_file.pdf")
            title = Path(source_name).stem

        # Extract text and render pages
        pages_text = []
        page_images = []
        page_count = len(doc)
        pages_to_process = page_count if max_pages is None else min(max_pages, page_count)

        for page_num in range(pages_to_process):
            page = doc[page_num]

            # Extract text
            text = page.get_text()
            has_text = bool(text.strip())
            if has_text:
                pages_text.append(text)

            # Check for images in page
            has_images = len(page.get_images()) > 0

            # Render page to image
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            pil_image = Image.open(BytesIO(img_data))

            page_images.append(
                PageImage(
                    page_num=page_num,
                    image=pil_image,
                    width=pix.width,
                    height=pix.height,
                    has_text=has_text,
                    has_images=has_images,
                )
            )

        doc.close()

        content = "\n\n".join(pages_text)

        metadata = {
            "format": "pdf",
            "page_count": page_count,
            "pages_with_text": len(pages_text),
            "pages_rendered": len(page_images),
            "render_dpi": dpi,
        }

        logger.info(
            f"Loaded PDF with images: {source_name} "
            f"({page_count} pages, {len(page_images)} images)"
        )

        return PDFWithImages(
            content=content,
            source=source_name,
            title=title,
            metadata=metadata,
            page_images=page_images,
        )

    def extract_page_images(
        self,
        source: str | Path | BinaryIO,
        page_numbers: list[int] | None = None,
        dpi: int | None = None,
    ) -> list[PageImage]:
        """Extract specific page images from a PDF.

        Args:
            source: Path to PDF file or file-like object
            page_numbers: Specific pages to extract (0-indexed), None = all
            dpi: Resolution for rendering

        Returns:
            List of PageImage objects
        """
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF required: pip install pymupdf")

        dpi = dpi or self.DEFAULT_DPI
        zoom = dpi / 72

        if isinstance(source, (str, Path)):
            doc = fitz.open(source)
        else:
            if hasattr(source, "read"):
                data = source.read()
                if isinstance(data, str):
                    data = data.encode("utf-8")
                doc = fitz.open(stream=data, filetype="pdf")
            else:
                doc = fitz.open(stream=source, filetype="pdf")

        page_images = []
        pages_to_extract = page_numbers if page_numbers else range(len(doc))

        for page_num in pages_to_extract:
            if page_num >= len(doc):
                continue

            page = doc[page_num]
            text = page.get_text()
            has_text = bool(text.strip())
            has_images = len(page.get_images()) > 0

            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            img_data = pix.tobytes("png")
            pil_image = Image.open(BytesIO(img_data))

            page_images.append(
                PageImage(
                    page_num=page_num,
                    image=pil_image,
                    width=pix.width,
                    height=pix.height,
                    has_text=has_text,
                    has_images=has_images,
                )
            )

        doc.close()
        return page_images
