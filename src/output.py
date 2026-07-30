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
from dataclasses import asdict, is_dataclass
from pathlib import Path


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
        text = p.text
        if text.strip():
            data[str(p.number)] = text
        k = 0
        for el in p.elements:
            if el.kind == "image" and el.crop is not None:
                k += 1
                el.crop.save(out / f"problem_{p.number}_image_{k}.png")
    _write_json(out / "problems.json", data)
    return len(problems)


def write_problem_coverage(exceptions, dest):
    """Write a per-problem coverage sidecar for downstream consumers.

    ``exceptions`` is normally the result of ``Series.coverage_exceptions``.
    Always writing the file (including an empty object) makes a parsed test's
    coverage contract explicit to importers such as Problem Cloud.
    """
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    data = {
        str(number): asdict(value) if is_dataclass(value) else dict(value)
        for number, value in sorted(exceptions.items())
    }
    _write_json(out / "problem_coverage.json", data)
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
                if text and text.strip():
                    items.append(text)
                    written += 1
            if items:
                data[str(number)] = items
        elif value and value.strip():
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
