"""PDF -> page images."""

from pathlib import Path

import pymupdf


def pdf_to_images(pdf_path, output_folder="./pdf_images", dpi=200, skip_page=None):
    """Convert each page of a PDF into a PNG image. Returns the list of paths.

    `skip_page`, if given, is called with each page's embedded PDF text (empty
    string for a scanned page with no text layer); pages for which it returns
    True are neither rendered nor included in the result. Lets a `Series` drop
    front matter / instructions pages without running OCR on them.
    """
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    paths = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        if skip_page is not None and skip_page(page.get_text()):
            continue
        pix = page.get_pixmap(dpi=dpi)
        image_path = out / f"page_{page_num + 1}.png"
        pix.save(image_path)
        paths.append(image_path)
    doc.close()
    return paths


def pdf_pages_to_images(pdf_path, output_folder, page_indices, dpi=200):
    """Render selected zero-based PDF pages, preserving source page numbers.

    Compendia can contain many independent tests in one PDF.  Keeping filenames
    as ``page_<source page>.png`` lets the pipeline pair each render with the
    correct born-digital PDF page for text-layer-aware figure filtering.
    """
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    paths = []
    try:
        for page_index in page_indices:
            if page_index < 0 or page_index >= len(doc):
                raise IndexError(
                    f"PDF page index {page_index} outside 0..{len(doc) - 1}: "
                    f"{pdf_path}"
                )
            pix = doc.load_page(page_index).get_pixmap(dpi=dpi)
            image_path = out / f"page_{page_index + 1}.png"
            pix.save(image_path)
            paths.append(image_path)
    finally:
        doc.close()
    return paths
