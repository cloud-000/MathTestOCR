"""PDF -> page images."""

from pathlib import Path

import pymupdf


def pdf_to_images(pdf_path, output_folder="./pdf_images", dpi=200):
    """Convert each page of a PDF into a PNG image. Returns the list of paths."""
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    paths = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        image_path = out / f"page_{page_num + 1}.png"
        pix.save(image_path)
        paths.append(image_path)
    doc.close()
    return paths
