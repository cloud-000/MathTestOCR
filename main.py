"""CLI entry point.

Examples:
    # Legacy single page -> flat out/
    python main.py parse m0/2025_9.png
    python main.py parse m0/2025_9.png --debug

    # Whole competition: discover tests, parse new ones into out/<series>/<test>/
    python main.py parse --series usamts --data-dir /path/to/usamts
    python main.py parse --series usamts --data-dir /path/to/usamts --force
    python main.py parse --series mathcounts --test 2025_state_sprint

    # Scrape solutions alongside the parsed problems
    python main.py solutions --series usamts --data-dir /path/to/usamts

    # PDF -> page images
    python main.py pdf /path/to/test.pdf --out m1
"""

import argparse
import tempfile
from pathlib import Path

from src import config, output
from src.annotate import draw_detections
from src.nanonets import NanonetsClient
from src.ocr import OCRModel
from src.ocr_cache import PARSE_CACHE, SOLUTION_CACHE, OCRCache
from src.pdf_io import pdf_to_images
from src.pipeline import (
    ocr_pages_markdown,
    process_image,
    process_image_nanonets,
    process_test,
)
from src.series import Test, get_series, series_names


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


def _open_engine(engine):
    """Construct the engine's model/client once (reused across a whole batch)."""
    return NanonetsClient() if engine == "nanonets" else OCRModel()


def _close_engine(engine, model):
    if engine == "mlx":
        model.unload()


def _resolve_data_dir(series, data_dir):
    dd = data_dir or config.SERIES_DATA_DIRS.get(series.name)
    if not dd:
        raise SystemExit(
            f"--data-dir is required for series '{series.name}' "
            f"(no default configured in config.SERIES_DATA_DIRS)"
        )
    return dd


def _select_tests(series, tests, test_id):
    """Filter discovered tests to an exact `--test` id, if given."""
    if test_id is None:
        return tests
    selected = [t for t in tests if t.id == test_id]
    if not selected:
        available = ", ".join(t.id for t in tests) or "(none)"
        raise SystemExit(
            f"[{series.name}] no test '{test_id}' found; available: {available}"
        )
    return selected


def _parse_one_test(series, test, engine, model, threshold, cache=None):
    """Render a test to pages and parse them into a merged, post-processed list."""
    match = series.match_marker()
    with tempfile.TemporaryDirectory(prefix="comp-ocr-") as workdir:
        pages = series.test_pages(test, workdir)
        problems = process_test(pages, engine, model, threshold, match, cache=cache)
    return series.postprocess(problems)


def _cmd_parse_batch(args):
    series = get_series(args.series)
    data_dir = _resolve_data_dir(series, args.data_dir)
    out_root = Path(args.out) / series.name

    tests = series.discover_tests(data_dir)
    print(f"[{series.name}] discovered {len(tests)} test(s) in {data_dir}")
    tests = _select_tests(series, tests, args.test)
    if not tests:
        return

    model = _open_engine(args.engine)
    try:
        for test in tests:
            dest = out_root / test.id
            if dest.exists() and not args.force:
                print(f"[{series.name}] skip {test.id} (already parsed; --force to redo)")
                continue
            print(f"[{series.name}] parsing {test.id} ...")
            cache = OCRCache(dest / PARSE_CACHE, enabled=args.cache)
            problems = _parse_one_test(
                series, test, args.engine, model, args.threshold, cache=cache
            )
            n = output.write_problems(problems, dest)
            print(f"[{series.name}] wrote {n} problem(s) -> {dest}")
    finally:
        _close_engine(args.engine, model)


def _cmd_parse_single(args):
    """Legacy single-page parse: flat output to --out, supports --debug overlay."""
    if args.engine == "nanonets":
        threshold = (
            args.threshold if args.threshold is not None else config.NANONETS_DETECT_THRESHOLD
        )
        problems, detections, groups = process_image_nanonets(
            args.image, NanonetsClient(), threshold
        )
        ocr = None
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

    n = output.write_problems(problems, args.out)
    print(f"\n[wrote {n} problem(s) -> {args.out}]")

    if ocr is not None:
        ocr.unload()


def cmd_parse(args):
    if args.series and args.image:
        raise SystemExit("pass either a single image OR --series, not both")
    if args.series:
        _cmd_parse_batch(args)
    elif args.image:
        _cmd_parse_single(args)
    else:
        raise SystemExit("provide an image path, or --series for batch parsing")


def cmd_solutions(args):
    series = get_series(args.series)
    if series.has_solutions:
        _cmd_solutions_ocr(args, series)
    elif series.has_answers:
        _cmd_answers(args, series)
    else:
        print(
            f"[{series.name}] has no solutions or answers (not yet supported); nothing to do"
        )


def _cmd_solutions_ocr(args, series):
    """OCR a per-test solution PDF into paired problem_<n>_solution.txt files."""
    data_dir = _resolve_data_dir(series, args.data_dir)
    out_root = Path(args.out) / series.name

    tests = series.discover_tests(data_dir)
    print(f"[{series.name}] discovered {len(tests)} test(s) in {data_dir}")
    tests = _select_tests(series, tests, args.test)
    match = series.match_marker()
    # A series can claim its whole solution document instead of the per-page
    # marker pipeline (see Series.parse_solutions). Only the nanonets engine
    # produces the whole-page markdown that path consumes.
    use_series_parser = series.custom_solution_parser and args.engine == "nanonets"

    model = _open_engine(args.engine)
    try:
        for test in tests:
            dest = out_root / test.id
            sol = series.solution_source(test)
            if sol is None:
                print(f"[{series.name}] skip {test.id} (no solution source found)")
                continue
            if any(dest.glob("problem_1_solution*.txt")) and not args.force:
                print(f"[{series.name}] skip {test.id} solutions (exist; --force to redo)")
                continue
            print(f"[{series.name}] scraping solutions for {test.id} ...")
            cache = OCRCache(dest / SOLUTION_CACHE, enabled=args.cache)
            with tempfile.TemporaryDirectory(prefix="comp-ocr-sol-") as workdir:
                pages = series.test_pages(Test(id=test.id, source=sol), workdir)
                if use_series_parser:
                    solutions = series.parse_solutions(
                        ocr_pages_markdown(pages, model, cache=cache)
                    )
                else:
                    problems = process_test(
                        pages, args.engine, model, args.threshold, match, cache=cache
                    )
                    problems = series.postprocess(problems)
                    solutions = {p.number: p.text for p in problems}
            n = output.write_solutions(solutions, dest)
            print(f"[{series.name}] wrote {n} solution file(s) -> {dest}")
    finally:
        _close_engine(args.engine, model)


def _cmd_answers(args, series):
    """Write an answer key (no OCR / no model) into problem_<n>_answer.txt files."""
    data_dir = _resolve_data_dir(series, args.data_dir)
    out_root = Path(args.out) / series.name

    tests = series.discover_tests(data_dir)
    print(f"[{series.name}] discovered {len(tests)} test(s) in {data_dir}")
    tests = _select_tests(series, tests, args.test)

    for test in tests:
        dest = out_root / test.id
        if (dest / f"problem_1_answer.txt").exists() and not args.force:
            print(f"[{series.name}] skip {test.id} answers (exist; --force to redo)")
            continue
        answers = series.scrape_answers(test)
        if not answers:
            print(f"[{series.name}] skip {test.id} (no answers found)")
            continue
        n = output.write_solutions(answers, dest, suffix="answer")
        print(f"[{series.name}] wrote {n} answer file(s) -> {dest}")


def cmd_pdf(args):
    paths = pdf_to_images(args.pdf, args.out)
    print(f"Wrote {len(paths)} page images to {args.out}")


def _add_engine_args(p):
    p.add_argument(
        "--engine",
        choices=["nanonets", "mlx"],
        default=config.DEFAULT_ENGINE,
        help="parsing engine (default: %(default)s)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="DETR detection confidence (default: engine-specific)",
    )
    p.add_argument(
        "--out",
        default=config.DEFAULT_OUT_DIR,
        help="output root (series mode writes <out>/<series>/<test>/)",
    )
    p.add_argument(
        "--cache",
        action="store_true",
        help="cache whole-page OCR in each test dir and reuse it on later runs "
        "(nanonets only; DETR still runs). Delete the cache file to re-OCR.",
    )


def main():
    parser = argparse.ArgumentParser(description="Parse competitive-math tests.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", help="parse a page image, or a whole series")
    p_parse.add_argument("image", nargs="?", help="single page image (legacy mode)")
    p_parse.add_argument(
        "--series", choices=series_names(), help="parse a whole competition series"
    )
    p_parse.add_argument("--data-dir", help="external source dir for the series")
    p_parse.add_argument(
        "--test", help="only parse this test id (exact match, e.g. 2025_state_sprint)"
    )
    p_parse.add_argument(
        "--force", action="store_true", help="re-parse tests even if already present"
    )
    p_parse.add_argument("--debug", action="store_true", help="save annotated overlay (single image)")
    _add_engine_args(p_parse)
    p_parse.set_defaults(func=cmd_parse)

    p_sol = sub.add_parser("solutions", help="scrape per-problem solutions for a series")
    p_sol.add_argument("--series", required=True, choices=series_names())
    p_sol.add_argument("--data-dir", help="external source dir for the series")
    p_sol.add_argument(
        "--test", help="only scrape this test id (exact match, e.g. 2025_state_sprint)"
    )
    p_sol.add_argument(
        "--force", action="store_true", help="re-scrape even if solution files exist"
    )
    _add_engine_args(p_sol)
    p_sol.set_defaults(func=cmd_solutions)

    p_pdf = sub.add_parser("pdf", help="convert a PDF to page images")
    p_pdf.add_argument("pdf")
    p_pdf.add_argument("--out", default="m0")
    p_pdf.set_defaults(func=cmd_pdf)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
