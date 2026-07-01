"""Write parsed problems and scraped solutions to a destination directory.

Layout inside `dest` (typically ``out/<series>/<test>/``):
  problem_<n>.txt                     -- statement text
  problem_<n>_image_<k>.png           -- 1-based image crop per problem
  problem_<n>_solution.txt            -- solution text (`_<k>.txt` when several)
  problem_<n>_solution_image_<k>.png  -- figure crops from the solution document
  problem_<n>_answer.txt              -- answer-key entry
"""

from pathlib import Path


def write_problems(problems, dest):
    """Write each problem's statement + image crops into `dest`.

    Returns the number of problems written.
    """
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    # Image crops are fully regenerated below; clear any from a prior parse so a
    # figure that moved to a different problem doesn't leave a stale crop behind
    # (e.g. an old problem_6_image_1.png after it's reassigned to problem 3).
    for stale in out.glob("problem_*_image_*.png"):
        stale.unlink()
    for p in problems:
        text = p.text
        if text.strip():
            (out / f"problem_{p.number}.txt").write_text(text)
        k = 0
        for el in p.elements:
            if el.kind == "image" and el.crop is not None:
                k += 1
                el.crop.save(out / f"problem_{p.number}_image_{k}.png")
    return len(problems)


def write_solutions(solutions, dest, suffix="solution"):
    """Write per-problem solution/answer text alongside the problems in `dest`.

    `solutions` maps problem number -> value, where a value is either a single
    string or a list of strings (a problem may have several distinct solutions).
    A string is written as ``problem_<n>_<suffix>.txt``; a list is written as
    ``problem_<n>_<suffix>_<k>.txt`` (1-based), so multiple solutions per problem
    are preserved. Empty entries are skipped. `dest` is created if missing.
    Returns the number of files written.
    """
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for number, value in solutions.items():
        if isinstance(value, (list, tuple)):
            for k, text in enumerate(value, start=1):
                if text and text.strip():
                    (out / f"problem_{number}_{suffix}_{k}.txt").write_text(text)
                    written += 1
        elif value and value.strip():
            (out / f"problem_{number}_{suffix}.txt").write_text(value)
            written += 1
    return written


def write_solution_images(figures, dest):
    """Write each problem's solution figure crops into `dest`.

    `figures` maps problem number -> list of PIL crops (see
    pipeline.process_solution_document). Crops are fully regenerated on every
    scrape: stale ``problem_*_solution_image_*.png`` from a prior run are
    removed first, mirroring `write_problems`. Returns the number written.
    """
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("problem_*_solution_image_*.png"):
        stale.unlink()
    written = 0
    for number, crops in figures.items():
        for k, crop in enumerate(crops, start=1):
            crop.save(out / f"problem_{number}_solution_image_{k}.png")
            written += 1
    return written
