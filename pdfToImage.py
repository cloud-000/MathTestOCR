import os

import pymupdf


def pdf_to_images(pdf_path, output_folder="./pdf_images"):
    """
    Converts each page of a PDF into a PNG image.

    Args:
        pdf_path (str): The path to the input PDF file.
        output_folder (str): The directory where the images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        doc = pymupdf.open(pdf_path)
        print(f"Processing PDF: {pdf_path}")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
            pix.save(image_path)
            print(f"Saved: {image_path}")
        doc.close()
        print(f"Conversion complete. Images saved to: {output_folder}")
    except Exception as e:
        print(f"An error occurred: {e}")
