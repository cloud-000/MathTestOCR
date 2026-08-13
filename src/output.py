"""Write parsed problems and scraped solutions to a destination directory.

Layout inside `dest` (typically ``out/<series>/<test>/``):
  problems.json                       -- {problem number: statement text}
  problem_<n>_image_<k>.png           -- 1-based image crop per problem
  problem_solution.json               -- {problem number: [solution text, ...]}
  problem_<n>_solution_<k>_image_<j>.png
                                     -- figure crops from the solution document
  problem_answer.json                 -- {problem number: answer-key entry}
  problem_coverage.json               -- verified non-standard coverage metadata
  test_profile.json                    -- proof-test semantics, when applicable
"""

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path


# OCR frequently leaves a Markdown rule or a lone emphasis delimiter at the
# edge of an otherwise-complete entry.  These lines carry no semantic content;
# importantly, the pattern is line-wide and is only applied at the *boundary*,
# so internal thematic breaks and real emphasis (``**Claim.**``) survive.
_BOUNDARY_MARKDOWN_RE = re.compile(
    r"^[ \t]*(?:(?:\*[ \t]*){2,}|(?:_[ \t]*){2,}|(?:-[ \t]*){3,})$"
)
_IMAGE_REF_LINE_RE = re.compile(r"^\s*!\[[^]]*\]\([^)]+\)\s*$")


def clean_output_boundary(text: str) -> str:
    """Remove content-free Markdown control lines from an entry's edges."""
    lines = text.splitlines()
    while lines and (not lines[0].strip() or _BOUNDARY_MARKDOWN_RE.fullmatch(lines[0])):
        lines.pop(0)
    while lines and (not lines[-1].strip() or _BOUNDARY_MARKDOWN_RE.fullmatch(lines[-1])):
        lines.pop()
    # Inline figure refs are inserted before serialization. Treat those refs as
    # transparent while locating the text boundary, so ``statement\n***\n![]``
    # loses the OCR rule without losing or moving its figure.
    semantic = [
        index
        for index, line in enumerate(lines)
        if line.strip() and not _IMAGE_REF_LINE_RE.fullmatch(line)
    ]
    while semantic and _BOUNDARY_MARKDOWN_RE.fullmatch(lines[semantic[-1]]):
        del lines[semantic[-1]]
        semantic = [
            index
            for index, line in enumerate(lines)
            if line.strip() and not _IMAGE_REF_LINE_RE.fullmatch(line)
        ]
    while semantic and _BOUNDARY_MARKDOWN_RE.fullmatch(lines[semantic[0]]):
        del lines[semantic[0]]
        semantic = [
            index
            for index, line in enumerate(lines)
            if line.strip() and not _IMAGE_REF_LINE_RE.fullmatch(line)
        ]
    cleaned = "\n".join(lines).strip()
    # Some OCR wraps the *entire* reconstructed entry in bold delimiters, or
    # emits only one of that pair. Limit inline cleanup to the outermost bytes;
    # ordinary internal emphasis remains untouched.
    if cleaned.startswith(("** ", "**\n")) and cleaned.endswith("**") and len(cleaned) > 4:
        cleaned = cleaned[2:-2].strip()
    else:
        if cleaned.startswith("** ") and cleaned.count("**") % 2:
            cleaned = cleaned[2:].lstrip()
        if cleaned.endswith(" **") and cleaned.count("**") % 2:
            cleaned = cleaned[:-2].rstrip()
    return cleaned


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _remove_legacy_problem_text(out):
    for stale in out.glob("problem_*.txt"):
        parts = stale.stem.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            stale.unlink()


def _remove_legacy_solution_text(out, suffix):
    for stale in out.glob(f"problem_*_{suffix}*.txt"):
        stale.unlink()


def write_problems(problems, dest):
    """Write each problem's statement + image crops into `dest`.

    Returns the number of problems written.
    """
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    _remove_legacy_problem_text(out)
    # Image crops are fully regenerated below; clear any from a prior parse so a
    # figure that moved to a different problem doesn't leave a stale crop behind
    # (e.g. an old problem_6_image_1.png after it's reassigned to problem 3).
    for stale in out.glob("problem_*_image_*.png"):
        # The broad statement-image pattern also matches
        # problem_<n>_solution_<k>_image_<j>.png. Those crops belong to a
        # separate regeneration pass and must survive statement-only parses.
        if "_solution_" not in stale.name:
            stale.unlink()
    data = {}
    for p in problems:
        text = clean_output_boundary(p.text)
        if text.strip():
            data[str(p.number)] = text
        k = 0
        for el in p.elements:
            if el.kind == "image" and el.crop is not None:
                k += 1
                el.crop.save(out / f"problem_{p.number}_image_{k}.png")
    _write_json(out / "problems.json", data)
    return len(problems)


def write_problem_texts(problems, dest):
    """Rewrite only ``problems.json``, preserving all existing image crops.

    Used by statement ``--reparse``: cached OCR can reconstruct text and image
    references, but DETR is deliberately not rerun and therefore supplies no
    crop objects to save. Calling ``write_problems`` there would correctly
    interpret the absent crops as a fresh parse and delete the old files.
    """
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for problem in problems:
        text = clean_output_boundary(problem.text)
        if text:
            data[str(problem.number)] = text
    _write_json(out / "problems.json", data)
    return len(data)


def write_problem_coverage(exceptions, dest):
    """Write a per-problem coverage sidecar for downstream consumers.

    ``exceptions`` is normally the result of ``Series.coverage_exceptions``.
    The sidecar is omitted when there are no exceptions.  An empty sidecar from
    a previous parse is removed so reruns leave the output tree consistent.
    """
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    data = {
        str(number): asdict(value) if is_dataclass(value) else dict(value)
        for number, value in sorted(exceptions.items())
    }
    path = out / "problem_coverage.json"
    if data:
        _write_json(path, data)
    elif path.exists():
        path.unlink()
    return len(data)


def write_test_profile(profile, dest):
    """Write (or clear) a test-level content profile for downstream importers."""
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "test_profile.json"
    if profile is None:
        if path.exists():
            path.unlink()
        return False
    _write_json(path, asdict(profile) if is_dataclass(profile) else dict(profile))
    return True


def write_solutions(solutions, dest, suffix="solution"):
    """Write per-problem solution/answer JSON alongside the problems in `dest`.

    `solutions` maps problem number -> value, where a value is either a single
    string or a list of strings (a problem may have several distinct solutions).
    Solutions are normalized to arrays in ``problem_solution.json``; answers are
    written as a string map in ``problem_answer.json``. Empty entries are
    skipped. `dest` is created if missing. Returns the number of entries written.
    """
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    _remove_legacy_solution_text(out, suffix)
    path = out / f"problem_{suffix}.json"
    data = {}
    written = 0
    for number, value in solutions.items():
        if isinstance(value, (list, tuple)):
            items = []
            for text in value:
                text = clean_output_boundary(text) if text else ""
                if text:
                    items.append(text)
                    written += 1
            if items:
                data[str(number)] = items
        elif value and (value := clean_output_boundary(value)):
            data[str(number)] = [value] if suffix == "solution" else value
            written += 1
    if data:
        _write_json(path, data)
    elif path.exists():
        path.unlink()
    return written


def write_solution_images(figures, dest):
    """Write each problem's solution figure crops into `dest`.

    `figures` maps problem number -> {solution index: list of PIL crops} (see
    pipeline.process_solution_document). Crops are fully regenerated on every
    scrape: stale solution image crops from a prior run are removed first,
    mirroring `write_problems`. Returns the number written.
    """
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("problem_*_solution_image_*.png"):
        stale.unlink()
    for stale in out.glob("problem_*_solution_*_image_*.png"):
        stale.unlink()
    written = 0
    for number, value in figures.items():
        if isinstance(value, dict):
            solution_groups = value.items()
        else:
            # Backward-compatible flat input: all crops belong to solution 1.
            solution_groups = [(1, value)]
        for solution, crops in solution_groups:
            for k, crop in enumerate(crops, start=1):
                crop.save(out / f"problem_{number}_solution_{solution}_image_{k}.png")
                written += 1
    return written
