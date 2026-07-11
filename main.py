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
from dataclasses import replace
from pathlib import Path

from src import config, output
from src.annotate import draw_detections
from src.nanonets import NanonetsClient
from src.ocr import OCRModel
from src.ocr_cache import PARSE_CACHE, SOLUTION_CACHE, OCRCache
from src.pdf_io import pdf_to_images
from src.pipeline import (
    inline_problem_figures,
    inline_solution_figures,
    ocr_pages,
    process_image,
    process_image_nanonets,
    process_solution_document,
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


def _filter_existing(series, tests, out_root, existing):
    """Keep only tests that already have an output folder under `out_root`.

    Used by `--existing` to re-run over the tests already parsed on disk instead
    of every test discovered in the data dir.
    """
    if not existing:
        return tests
    selected = [t for t in tests if (Path(out_root) / t.id).is_dir()]
    print(f"[{series.name}] --existing: {len(selected)} of {len(tests)} test(s) present in {out_root}")
    return selected


def _resolve_layout(series, args):
    """Series LayoutOptions with the CLI ``--temp`` override applied (CLI wins).

    ``--temp`` sets the base OCR temperature (rung 0 of the runaway-recovery
    ladder), overriding whatever the series' LayoutOptions specifies -- the
    manual escape hatch for a test whose grids loop even after auto-escalation.
    """
    layout = series.layout_options()
    if getattr(args, "temp", None) is not None:
        layout = replace(layout, nanonets_temperature=args.temp)
    return layout


def _parse_one_test(series, test, engine, model, threshold, cache=None, layout=None):
    """Render a test to pages and parse them into a merged, post-processed list."""
    match = series.match_marker()
    layout = layout if layout is not None else series.layout_options()
    with tempfile.TemporaryDirectory(prefix="comp-ocr-") as workdir:
        pages = series.test_pages(test, workdir)
        problems = process_test(
            pages, engine, model, threshold, match, cache=cache, layout=layout
        )
    return series.postprocess(problems)


def _cmd_parse_batch(args):
    series = get_series(args.series)
    data_dir = _resolve_data_dir(series, args.data_dir)
    out_root = Path(args.out) / series.name

    tests = series.discover_tests(data_dir)
    print(f"[{series.name}] discovered {len(tests)} test(s) in {data_dir}")
    tests = _select_tests(series, tests, args.test)
    tests = _filter_existing(series, tests, out_root, args.existing)
    if not tests:
        return

    layout = _resolve_layout(series, args)
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
                series, test, args.engine, model, args.threshold, cache=cache, layout=layout
            )
            if layout.inline_figures:
                inline_problem_figures(problems, f"{series.name}/{test.id}/")
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
        layout = (
            config.LayoutOptions(nanonets_temperature=args.temp)
            if args.temp is not None
            else None
        )
        problems, detections, groups = process_image_nanonets(
            args.image, NanonetsClient(), threshold, layout=layout
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
    """Scrape per-problem solutions (text + figure crops) and/or answer keys."""
    series = get_series(args.series)
    if not (series.has_solutions or series.has_answers):
        print(
            f"[{series.name}] has no solutions or answers (not yet supported); nothing to do"
        )
        return
    data_dir = _resolve_data_dir(series, args.data_dir)
    out_root = Path(args.out) / series.name

    tests = series.discover_tests(data_dir)
    print(f"[{series.name}] discovered {len(tests)} test(s) in {data_dir}")
    tests = _select_tests(series, tests, args.test)
    tests = _filter_existing(series, tests, out_root, args.existing)
    if not tests:
        return

    model = _open_engine(args.engine)
    try:
        for test in tests:
            dest = out_root / test.id
            sol = series.solution_source(test) if series.has_solutions else None
            pages_md = None
            if series.has_solutions and sol is None:
                print(f"[{series.name}] skip {test.id} (no solution source found)")
            if sol is not None:
                pages_md = _scrape_solutions(args, series, test, sol, dest, model)
            if series.has_answers:
                _scrape_answers(
                    args, series, test, dest, model, out_root, data_dir, sol, pages_md
                )
    finally:
        _close_engine(args.engine, model)


def _scrape_solutions(args, series, test, sol, dest, model):
    """OCR one test's solution document into solution text + figure crops.

    Returns the raw per-page markdown when the OCR actually ran, so a same-file
    answer key (see Series.answer_source) can be parsed without re-OCR; None
    when the test was skipped or the mlx engine (no whole-page markdown) ran.
    """
    if (dest / "problem_solution.json").exists() and not args.force:
        print(f"[{series.name}] skip {test.id} solutions (exist; --force to redo)")
        return None
    print(f"[{series.name}] scraping solutions for {test.id} ...")
    cache = OCRCache(dest / SOLUTION_CACHE, enabled=args.cache)
    pages_md = None
    with tempfile.TemporaryDirectory(prefix="comp-ocr-sol-") as workdir:
        pages = series.test_pages(Test(id=test.id, source=sol), workdir)
        if args.engine == "nanonets":
            pages_md, figures = process_solution_document(
                pages,
                model,
                args.threshold,
                match=series.match_marker(),
                cache=cache,
                layout=_resolve_layout(series, args),
                clean_page=series.clean_solution_markdown,
                source_pdf=sol,
                match_solution=series.solution_index_marker,
            )
            cleaned = "\n\n".join(
                series.clean_solution_markdown(i, md) for i, md in enumerate(pages_md)
            )
            solutions = series.parse_solutions(cleaned)
        else:
            # Legacy mlx path: per-page marker pipeline; figures come from the
            # problems' own image elements.
            problems = series.postprocess(
                process_test(
                    pages, args.engine, model, args.threshold, series.match_marker(),
                    cache=cache, layout=_resolve_layout(series, args),
                )
            )
            solutions = {p.number: p.text for p in problems}
            figures = {
                p.number: {
                    1: [
                        el.crop
                        for el in p.elements
                        if el.kind == "image" and el.crop is not None
                    ]
                }
                for p in problems
            }
    # Place each figure crop inline in its solution text. Crops are referenced
    # by a path relative to the output root (out/<series>/<test>/), so the marker
    # resolves when out/ is served as the document root.
    path_prefix = f"{series.name}/{test.id}/"
    solutions = inline_solution_figures(solutions, figures, path_prefix)
    n = output.write_solutions(solutions, dest)
    k = output.write_solution_images(figures, dest)
    print(f"[{series.name}] wrote {n} solution(s), {k} figure crop(s) -> {dest}")
    return pages_md


def _scrape_answers(args, series, test, dest, model, out_root, data_dir, sol, sol_pages_md):
    """Write one test's answer key into problem_answer.json.

    Tries the no-OCR `scrape_answers` hook first (pre-scraped keys); otherwise
    OCRs `answer_source` and hands the pages to `parse_answers`. When the key
    lives inside the solution document just OCR'd, its markdown is reused.
    """
    if (dest / "problem_answer.json").exists() and not args.force:
        print(f"[{series.name}] skip {test.id} answers (exist; --force to redo)")
        return
    answers = series.scrape_answers(test)
    if not answers:
        src = series.answer_source(test)
        if src is None:
            print(f"[{series.name}] skip {test.id} answers (no answer source found)")
            return
        if args.engine != "nanonets":
            print(f"[{series.name}] skip {test.id} answers (need the nanonets engine)")
            return
        if sol_pages_md is not None and src == sol:
            pages_md = sol_pages_md
        else:
            # A same-file key shares the test's solution cache; a standalone
            # answer document gets one cache shared across every test it covers
            # (Mathcounts: one answers.pdf serves sprint/target/team/...).
            if src == sol:
                cache = OCRCache(dest / SOLUTION_CACHE, enabled=args.cache)
            else:
                cache = OCRCache(
                    _answer_cache_path(out_root, data_dir, src), enabled=args.cache
                )
            with tempfile.TemporaryDirectory(prefix="comp-ocr-ans-") as workdir:
                pages = series.test_pages(Test(id=test.id, source=src), workdir)
                pages_md = ocr_pages(pages, model, cache=cache, layout=_resolve_layout(series, args))
        answers = series.parse_answers(test, pages_md)
    if not answers:
        print(f"[{series.name}] skip {test.id} (no answers found)")
        return
    n = output.write_solutions(answers, dest, suffix="answer")
    print(f"[{series.name}] wrote {n} answer(s) -> {dest}")


def _answer_cache_path(out_root, data_dir, src):
    """One shared OCR-cache file per answer document, under out/<series>/.

    Named by the document's path relative to the data dir so the several tests
    that share one key (e.g. every Mathcounts round in a level) hit one cache.
    """
    src = Path(src).resolve()
    try:
        stem = "_".join(src.relative_to(Path(data_dir).resolve()).with_suffix("").parts)
    except ValueError:
        stem = src.stem
    return Path(out_root) / "_answers_ocr" / f"{stem}.json"


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
        "--temp",
        type=float,
        default=None,
        help="base nanonets OCR temperature, overriding the series default "
        "(rung 0 of the runaway-recovery ladder). Raise for a test whose grids "
        "still loop after auto-escalation, e.g. --temp 0.4.",
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
    p_parse.add_argument(
        "--existing",
        action="store_true",
        help="only process tests that already have an out/<series>/<test>/ folder",
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
        "--force", action="store_true", help="re-scrape even if solution JSON exists"
    )
    p_sol.add_argument(
        "--existing",
        action="store_true",
        help="only process tests that already have an out/<series>/<test>/ folder",
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
