"""One-time script to generate test fixture files."""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


FIXTURES_DIR = Path(__file__).parent


def generate_sample_pdf():
    """Create a 2-page PDF with known text content."""
    path = FIXTURES_DIR / "sample.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)

    # Page 1
    c.drawString(72, 720, "Page 1: Introduction to Semantic Chunking")
    c.drawString(72, 700, "This paper presents a novel approach to text segmentation.")
    c.drawString(72, 680, "We propose using embedding-based similarity for chunk boundaries.")
    c.showPage()

    # Page 2
    c.drawString(72, 720, "Page 2: Results and Discussion")
    c.drawString(72, 700, "Our method achieves state-of-the-art performance on three benchmarks.")
    c.drawString(72, 680, "The semantic chunker preserves paragraph-level coherence.")
    c.showPage()

    c.save()
    print(f"Generated: {path}")


def generate_corrupt_file():
    """Create a non-PDF binary file."""
    path = FIXTURES_DIR / "corrupt.bin"
    path.write_bytes(b"not a pdf\x00\xff\xfe\xfd")
    print(f"Generated: {path}")


if __name__ == "__main__":
    generate_sample_pdf()
    generate_corrupt_file()
