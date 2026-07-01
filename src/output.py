"""Write parsed problems and scraped solutions to a destination directory.

Layout inside `dest` (typically ``out/<series>/<test>/``):
  problem_<n>.txt            -- statement text
  problem_<n>_image_<k>.png  -- 1-based image crop per problem
  problem_<n>_solution.txt   -- solution text (written by the solutions command)
  problem_<n>_answer.txt     -- answer (series with answers but no full solutions)
"""

from pathlib import Path


def write_problems(problems, dest):
    """Write each problem's statement + image crops into `dest`.

    Returns the number of problems written.
    """
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
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

    `solutions` maps problem number -> text. Files are named
    ``problem_<n>_<suffix>.txt`` (suffix "solution" or "answer"). Empty entries
    are skipped. `dest` is expected to already hold the parsed problems, but is
    created if missing. Returns the number of files written.
    """
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for number, text in solutions.items():
        if text and text.strip():
            (out / f"problem_{number}_{suffix}.txt").write_text(text)
            written += 1
    return written
