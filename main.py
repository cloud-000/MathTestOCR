"""CLI entry point.

Examples:
    python main.py parse m0/2025_9.png
    python main.py parse m0/2025_9.png --debug
    python main.py pdf /path/to/test.pdf --out m1
"""

import argparse
from pathlib import Path

from src import config
from src.annotate import draw_detections
from src.nanonets import NanonetsClient
from src.ocr import OCRModel
from src.pdf_io import pdf_to_images
from src.pipeline import process_image, process_image_nanonets


def _print_problems(problems):
    for p in problems:
        print(f"\n# ===== Problem {p.number} ===== #")
        for el in p.elements:
            if el.kind == "text":
                print(el.text)
            elif el.box:
                print(f"[{el.label} image @ {el.box}]")
            else:
                # Figure Nanonets detected but DETR did not crop.
                print(f"[{el.label} image (no crop): {el.text}]")


def cmd_parse(args):
    ocr = None
    if args.engine == "nanonets":
        threshold = args.threshold if args.threshold is not None else config.NANONETS_DETECT_THRESHOLD
        problems, detections, groups = process_image_nanonets(
            args.image, NanonetsClient(), threshold
        )
    else:
        threshold = args.threshold if args.threshold is not None else config.DETECT_THRESHOLD
        ocr = OCRModel()
        problems, detections, groups = process_image(args.image, ocr, threshold)
    _print_problems(problems)

    if args.debug:
        from PIL import Image

        debug_dir = Path(config.DEFAULT_DEBUG_DIR)
        debug_dir.mkdir(parents=True, exist_ok=True)
        img = Image.open(args.image).convert("RGB")
        out = debug_dir / f"{Path(args.image).stem}_annotated.png"
        draw_detections(img, detections, groups).save(out)
        print(f"\n[debug overlay -> {out}]")

    if args.save_images:
        out_dir = Path(args.save_images)
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in problems:
            for i, el in enumerate(p.elements):
                if el.kind == "image" and el.crop is not None:
                    el.crop.save(out_dir / f"p{p.number}_img{i}.png")

    if ocr is not None:
        ocr.unload()


def cmd_pdf(args):
    paths = pdf_to_images(args.pdf, args.out)
    print(f"Wrote {len(paths)} page images to {args.out}")


def main():
    parser = argparse.ArgumentParser(description="Parse competitive-math tests.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", help="parse a single page image")
    p_parse.add_argument("image")
    p_parse.add_argument(
        "--engine",
        choices=["nanonets", "mlx"],
        default=config.DEFAULT_ENGINE,
        help="parsing engine (default: %(default)s)",
    )
    p_parse.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="DETR detection confidence (default: engine-specific)",
    )
    p_parse.add_argument("--debug", action="store_true", help="save annotated overlay")
    p_parse.add_argument("--save-images", help="dir to save problem image crops")
    p_parse.set_defaults(func=cmd_parse)

    p_pdf = sub.add_parser("pdf", help="convert a PDF to page images")
    p_pdf.add_argument("pdf")
    p_pdf.add_argument("--out", default="m0")
    p_pdf.set_defaults(func=cmd_pdf)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
