"""CLI entry point.

Examples:
    # Legacy single page -> flat out/
    python main.py parse m0/2025_9.png
    python main.py parse m0/2025_9.png --debug

    # Whole competition: discover tests, parse new ones into out/<series>/<test>/
    python main.py parse --series usamts --data-dir /path/to/usamts
    python main.py parse --series usamts --data-dir /path/to/usamts --force
    python main.py parse --series mathcounts --test 2025_state_sprint

    # Unregistered competition: one PDF/image folder, or a directory of PDFs
    python main.py parse-series custom-name /path/to/source --out out

    # Scrape solutions alongside the parsed problems
    python main.py solutions --series usamts --data-dir /path/to/usamts

    # PDF -> page images
    python main.py pdf /path/to/test.pdf --out m1
"""

import argparse
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from src import config, output
from src.annotate import draw_detections
from src.llama import LlamaClient
from src.nanonets import NanonetsClient
from src.ocr import OCRModel
from src.ocr_cache import PARSE_CACHE, SOLUTION_CACHE, OCRCache
from src.pdf_io import pdf_to_images
from src.pipeline import (
    inline_problem_figures,
    inline_solution_figures,
    ocr_pages,
    process_image,
    process_image_markdown,
    process_solution_document,
    process_test,
)
from src.series import GenericSeries, Test, get_series, series_names


class SummaryTracker:
    def __init__(self):
        self.enabled = False
        self.events = []
        self._stdout = None
        self._stderr = None
        self._devnull = None

    def start(self, enabled=True):
        self.enabled = enabled
        self.events = []
        if self.enabled:
            self._stdout = sys.stdout
            self._stderr = sys.stderr
            self._devnull = open(os.devnull, "w")
            sys.stdout = self._devnull
            sys.stderr = self._devnull

    def log(self, msg=""):
        msg_str = str(msg)
        if self.enabled:
            self.events.append(msg_str)
        else:
            print(msg_str)

    def stop(self, status=None):
        if not self.enabled:
            return
        if self._devnull is not None:
            sys.stdout = self._stdout
            sys.stderr = self._stderr
            self._devnull.close()
            self._devnull = None

        self._print_summary(status)
        self.enabled = False

    def _print_summary(self, status=None):
        if status:
            header = f"\nSummary of completed work ({status}):"
        else:
            header = "\nSummary of completed work:"
        print(header)
        if not self.events:
            print("  (no actions performed)")
        else:
            for item in self.events:
                for line in item.splitlines():
                    if line.strip():
                        print(f"  {line}")


_tracker = SummaryTracker()


def log(msg=""):
    _tracker.log(msg)


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


def _open_engine(engine, llama_tier=None):
    """Construct the engine's model/client once (reused across a whole batch).

    `llama_tier` overrides config.LLAMA_TIER for the llama engine (the CLI
    ``--llama-tier`` flag); ignored by the other engines.
    """
    if engine == "nanonets":
        return NanonetsClient()
    if engine == "llama":
        return LlamaClient(tier=llama_tier or config.LLAMA_TIER)
    return OCRModel()


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


def _exclude_ignored_tests(series, tests):
    """Apply permanent series exclusions before selection or OCR startup."""
    kept = []
    ignored = []
    for test in tests:
        (ignored if series.ignore_test(test) else kept).append(test)
    if ignored:
        ids = ", ".join(test.id for test in ignored)
        log(f"[{series.name}] ignored {len(ignored)} test(s): {ids}")
    return kept


def _count_parsed(out_root, tests):
    """Count tests that have parsed problems and parsed solutions/answers."""
    test_count = 0
    solution_count = 0
    for t in tests:
        dest = Path(out_root) / t.id
        if (dest / "problems.json").exists():
            test_count += 1
        if (
            (dest / "problem_solution.json").exists()
            or (dest / "solutions.json").exists()
            or (dest / "problem_answer.json").exists()
        ):
            solution_count += 1
    return test_count, solution_count


def _filter_existing(series, tests, out_root, existing):
    """Keep only tests that already have an output folder under `out_root`.

    Used by `--existing` to re-run over the tests already parsed on disk instead
    of every test discovered in the data dir.
    """
    if not existing:
        return tests
    selected = [t for t in tests if (Path(out_root) / t.id).is_dir()]
    test_count, solution_count = _count_parsed(out_root, tests)
    log(
        f"[{series.name}] --existing: {len(selected)} of {len(tests)} test(s) present in {out_root} "
        f"(test-count: {test_count}, solution-count: {solution_count})"
    )
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
            pages,
            engine,
            model,
            threshold,
            match,
            cache=cache,
            layout=layout,
            clean_page=series.clean_statement_markdown,
            validate_page=series.validate_statement_markdown,
        )
    return series.postprocess(problems)


def _cmd_parse_batch(args):
    series = get_series(args.series)
    data_dir = _resolve_data_dir(series, args.data_dir)
    tests = series.discover_tests(data_dir)
    log(f"[{series.name}] discovered {len(tests)} test(s) in {data_dir}")
    tests = _exclude_ignored_tests(series, tests)
    tests = _select_tests(series, tests, args.test)
    _run_parse_batch(series, tests, args)


def _run_parse_batch(series, tests, args):
    """Parse an already-discovered set of tests with shared batch behavior."""
    out_root = Path(args.out) / series.name
    tests = _filter_existing(
        series, tests, out_root, getattr(args, "existing", False)
    )
    if not tests:
        return

    layout = _resolve_layout(series, args)
    model = _open_engine(args.engine, args.llama_tier)
    try:
        for test in tests:
            dest = out_root / test.id
            if (dest / "problems.json").exists() and not args.force:
                log(f"[{series.name}] skip {test.id} (already parsed; --force to redo)")
                continue
            log(f"[{series.name}] parsing {test.id} ...")
            cache = OCRCache(dest / PARSE_CACHE, enabled=args.cache)
            problems = _parse_one_test(
                series, test, args.engine, model, args.threshold, cache=cache, layout=layout
            )
            # Reference every figure crop from the statement text so a problem's
            # images are discoverable from problems.json alone, not only by
            # globbing crop files. With inline_figures the refs land at the
            # model's <img> positions; without it (no placeholders emitted) they
            # append at the end -- both handled by _place_figure_refs.
            inline_problem_figures(problems, f"{series.name}/{test.id}/")
            n = output.write_problems(problems, dest)
            log(f"[{series.name}] wrote {n} problem(s) -> {dest}")
    finally:
        _close_engine(args.engine, model)


def cmd_parse_series(args):
    """Parse an unregistered series from a PDF, image folder, or PDF directory."""
    try:
        series = GenericSeries(args.name)
        tests = series.discover_source(args.source, args.test_name)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    log(f"[{series.name}] discovered {len(tests)} test(s) in {args.source}")
    _run_parse_batch(series, tests, args)


def _cmd_parse_single(args):
    """Legacy single-page parse: flat output to --out, supports --debug overlay."""
    if args.engine in config.MARKDOWN_ENGINES:
        threshold = (
            args.threshold if args.threshold is not None else config.NANONETS_DETECT_THRESHOLD
        )
        layout = (
            config.LayoutOptions(nanonets_temperature=args.temp)
            if args.temp is not None
            else None
        )
        # --cache reuses the whole-page OCR across runs (keyed by image filename),
        # so repeated single-page parses don't re-hit a paid/slow endpoint -- the
        # same OCRCache the batch/solution paths use, here rooted at --out. DETR
        # still re-runs every time.
        cache = OCRCache(Path(args.out) / PARSE_CACHE, enabled=args.cache)
        problems, detections, groups = process_image_markdown(
            args.image, _open_engine(args.engine, args.llama_tier), threshold,
            cache=cache, layout=layout
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
        log(f"[debug overlay -> {out}]")

    n = output.write_problems(problems, args.out)
    log(f"[wrote {n} problem(s) -> {args.out}]")

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
        log(
            f"[{series.name}] has no solutions or answers (not yet supported); nothing to do"
        )
        return
    data_dir = _resolve_data_dir(series, args.data_dir)
    out_root = Path(args.out) / series.name

    tests = series.discover_tests(data_dir)
    log(f"[{series.name}] discovered {len(tests)} test(s) in {data_dir}")
    tests = _exclude_ignored_tests(series, tests)
    tests = _select_tests(series, tests, args.test)
    tests = _filter_existing(series, tests, out_root, args.existing)
    if not tests:
        return

    model = _open_engine(args.engine, args.llama_tier)
    try:
        for test in tests:
            dest = out_root / test.id
            sol = series.solution_source(test) if series.has_solutions else None
            pages_md = None
            if series.has_solutions and sol is None:
                log(f"[{series.name}] skip {test.id} (no solution source found)")
            if sol is not None:
                pages_md = _scrape_solutions(args, series, test, sol, dest, model)
            if series.has_answers:
                _scrape_answers(
                    args, series, test, dest, model, out_root, data_dir, sol, pages_md
                )
    finally:
        _close_engine(args.engine, model)


def _has_nonempty_json(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        return bool(data)
    except (OSError, ValueError):
        return False


def _scrape_solutions(args, series, test, sol, dest, model):
    """OCR one test's solution document into solution text + figure crops.

    Returns the raw per-page markdown when the OCR actually ran, so a same-file
    answer key (see Series.answer_source) can be parsed without re-OCR; None
    when the test was skipped or the mlx engine (no whole-page markdown) ran.
    """
    has_existing_solution = (
        _has_nonempty_json(dest / "problem_solution.json")
        or _has_nonempty_json(dest / "solutions.json")
    )
    if has_existing_solution and not args.force:
        log(f"[{series.name}] skip {test.id} solutions (exist; --force to redo)")
        return None
    log(f"[{series.name}] scraping solutions for {test.id} ...")
    cache = OCRCache(dest / SOLUTION_CACHE, enabled=args.cache)
    pages_md = None
    with tempfile.TemporaryDirectory(prefix="comp-ocr-sol-") as workdir:
        pages = series.test_pages(Test(id=test.id, source=sol), workdir)
        if args.engine in config.MARKDOWN_ENGINES:
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
                figure_floor=series.solution_figure_floor,
            )
            cleaned = "\n\n".join(
                series.clean_solution_markdown(i, md) for i, md in enumerate(pages_md)
            )
            solutions = series.parse_solutions(cleaned, test=test)
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
    statements = {}
    statement_path = dest / "problems.json"
    if statement_path.exists():
        try:
            statements = json.loads(statement_path.read_text())
        except (OSError, ValueError):
            statements = {}
    solutions = series.postprocess_solutions(solutions, statements)
    # Place each figure crop inline in its solution text. Crops are referenced
    # by a path relative to the output root (out/<series>/<test>/), so the marker
    # resolves when out/ is served as the document root.
    path_prefix = f"{series.name}/{test.id}/"
    solutions = inline_solution_figures(solutions, figures, path_prefix)
    n = output.write_solutions(solutions, dest)
    k = output.write_solution_images(figures, dest)
    log(f"[{series.name}] wrote {n} solution(s), {k} figure crop(s) -> {dest}")
    return pages_md


def _scrape_answers(args, series, test, dest, model, out_root, data_dir, sol, sol_pages_md):
    """Write one test's answer key into problem_answer.json.

    Tries the no-OCR `scrape_answers` hook first (pre-scraped keys); otherwise
    OCRs `answer_source` and hands the pages to `parse_answers`. When the key
    lives inside the solution document just OCR'd, its markdown is reused.
    """
    has_existing_answer = _has_nonempty_json(dest / "problem_answer.json")
    if has_existing_answer and not args.force:
        log(f"[{series.name}] skip {test.id} answers (exist; --force to redo)")
        return
    answers = series.scrape_answers(test)
    if not answers:
        src = series.answer_source(test)
        if src is None:
            log(f"[{series.name}] skip {test.id} answers (no answer source found)")
            return
        if args.engine not in config.MARKDOWN_ENGINES:
            log(f"[{series.name}] skip {test.id} answers (need a markdown engine)")
            return
        if sol_pages_md is not None and src == sol:
            pages_md = sol_pages_md
        else:
            # A same-file key shares the test's solution cache; a standalone
            # answer document gets one cache shared across every test it covers
            # (Mathcounts: one answers.pdf serves sprint/target/team/...).
            if src == test.source:
                # Some presentation-style tests contain their own answer slides
                # (CHMM Integration Bee finals). Reuse the paid statement OCR.
                cache = OCRCache(dest / PARSE_CACHE, enabled=args.cache)
            elif src == sol:
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
        # A forced refresh is authoritative. Do not leave a stale partial key
        # behind when the current parser correctly finds that a proof round (or
        # an incomplete source packet) has no extractable short answers.
        if args.force:
            output.write_solutions({}, dest, suffix="answer")
        log(f"[{series.name}] skip {test.id} (no answers found)")
        return
    n = output.write_solutions(answers, dest, suffix="answer")
    log(f"[{series.name}] wrote {n} answer(s) -> {dest}")


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


def _preview(text, limit=160):
    """A single-line, length-capped snippet of a statement, for the manifest."""
    s = " ".join(text.split())
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def cmd_dedup(args):
    """Record problems shared across a series' tests into <series>/duplicates.json.

    Reads each already-parsed test's problems.json, buckets problems by the
    series' `duplicate_scope`, and groups near-duplicate statements (see
    `src.dedup`). Non-invasive: the per-test output is untouched; only the
    central manifest is written.
    """
    from src import dedup

    series = get_series(args.series)
    out_root = Path(args.out) / series.name
    if not out_root.is_dir():
        raise SystemExit(
            f"[{series.name}] no output dir {out_root}; parse the series first"
        )
    threshold = args.threshold if args.threshold is not None else config.DEDUP_THRESHOLD

    entries = []
    texts = {}
    scoped_tests = 0
    for test_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        pj = test_dir / "problems.json"
        if not pj.exists():
            continue
        scope = series.duplicate_scope(test_dir.name, across=args.across_years)
        if scope is None:
            continue
        scoped_tests += 1
        problems = json.loads(pj.read_text())
        for num, text in problems.items():
            ref = dedup.ProblemRef(test=test_dir.name, problem=str(num))
            entries.append((scope, ref, text))
            texts[ref] = text

    if scoped_tests == 0:
        log(
            f"[{series.name}] no parsed test defines a duplicate_scope; "
            "nothing to compare"
        )
        return

    groups = dedup.find_duplicate_groups(
        entries,
        threshold=threshold,
        k=config.DEDUP_SHINGLE_K,
        min_shingle_len=config.DEDUP_MIN_SHINGLE_LEN,
    )

    manifest = {
        "series": series.name,
        "threshold": threshold,
        "shingle_k": config.DEDUP_SHINGLE_K,
        "groups": [
            {
                "group": i + 1,
                "scope": g.scope,
                "similarity": g.similarity,
                "preview": _preview(texts[g.members[0]]),
                "members": [
                    {"test": m.test, "problem": m.problem} for m in g.members
                ],
            }
            for i, g in enumerate(groups)
        ],
    }
    dest = out_root / "duplicates.json"
    dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    dup_problems = sum(len(g.members) for g in groups)
    log(
        f"[{series.name}] {len(groups)} duplicate group(s) covering "
        f"{dup_problems} problem(s) across {scoped_tests} test(s) -> {dest}"
    )


def cmd_pdf(args):
    paths = pdf_to_images(args.pdf, args.out)
    log(f"Wrote {len(paths)} page images to {args.out}")


def _add_engine_args(p):
    p.add_argument(
        "--engine",
        choices=["nanonets", "llama", "mlx"],
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
        "--llama-tier",
        choices=list(config.LLAMA_TIERS),
        default=None,
        help="LlamaCloud parsing tier for --engine llama, quality/cost ascending "
        f"(default: {config.LLAMA_TIER}). Ignored by other engines.",
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
    p.add_argument(
        "--print-time",
        action="store_true",
        help="print the current time every time we parse a new page",
    )


def main():
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--no-print",
        action="store_true",
        help="suppress live output and only print a summary of completed work upon exit or interrupt",
    )

    parser = argparse.ArgumentParser(
        parents=[common_parser], description="Parse competitive-math tests."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser(
        "parse", parents=[common_parser], help="parse a page image, or a whole series"
    )
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

    p_generic = sub.add_parser(
        "parse-series", parents=[common_parser], help="parse an unregistered series from a PDF or directory"
    )
    p_generic.add_argument("name", help="custom series name used in the output path")
    p_generic.add_argument(
        "source", help="PDF, page-image folder, or directory of PDFs"
    )
    p_generic.add_argument(
        "--test-name", help="override the inferred test name for a single-test source"
    )
    p_generic.add_argument(
        "--force", action="store_true", help="re-parse tests even if already present"
    )
    p_generic.add_argument(
        "--existing",
        action="store_true",
        help="only process tests that already have an out/<series>/<test>/ folder",
    )
    _add_engine_args(p_generic)
    p_generic.set_defaults(func=cmd_parse_series)

    p_sol = sub.add_parser(
        "solutions", parents=[common_parser], help="scrape per-problem solutions for a series"
    )
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

    p_dedup = sub.add_parser(
        "dedup", parents=[common_parser], help="record problems shared across a series' tests"
    )
    p_dedup.add_argument("--series", required=True, choices=series_names())
    p_dedup.add_argument(
        "--out",
        default=config.DEFAULT_OUT_DIR,
        help="output root holding <series>/<test>/ (default: %(default)s)",
    )
    p_dedup.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Jaccard similarity cutoff (default: {config.DEDUP_THRESHOLD})",
    )
    p_dedup.add_argument(
        "--across-years",
        action="store_true",
        help="compare every test together instead of only within each year "
        "(catches problems recycled across years)",
    )
    p_dedup.set_defaults(func=cmd_dedup)

    p_pdf = sub.add_parser("pdf", parents=[common_parser], help="convert a PDF to page images")
    p_pdf.add_argument("pdf")
    p_pdf.add_argument("--out", default="m0")
    p_pdf.set_defaults(func=cmd_pdf)

    args = parser.parse_args()
    config.PRINT_TIME = getattr(args, "print_time", False)

    no_print = getattr(args, "no_print", False) or ("--no-print" in sys.argv)
    _tracker.start(enabled=no_print)
    try:
        args.func(args)
    except KeyboardInterrupt:
        _tracker.stop(status="interrupted")
        sys.exit(130)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
        _tracker.stop(status=None if code == 0 else f"exited with code {code}")
        raise
    except Exception as exc:
        _tracker.stop(status=f"error: {exc}")
        raise
    else:
        _tracker.stop()


if __name__ == "__main__":
    main()
