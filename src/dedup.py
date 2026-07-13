"""Cross-test duplicate detection.

Some series reuse the same problem across sibling tests -- PUMaC shares problems
between its A and B divisions of the same year (and, occasionally, across the
subject rounds within a year). This module finds near-duplicate problem
statements *within a comparison scope* and groups them, so the linkage can be
recorded without disturbing any per-test output.

Pure and series-agnostic: it takes already-parsed statements, each tagged with a
scope key (from `Series.duplicate_scope`), and returns duplicate groups. It
loads no model and does no I/O -- the caller reads `problems.json` and writes the
manifest (see `main.py::cmd_dedup`).

Matching. A statement is normalized to an alphanumeric word stream (`normalize`),
then reduced to a set of character k-gram shingles. Two statements are duplicates
when their shingle Jaccard similarity clears a threshold -- robust to the
punctuation/spacing/formatting drift OCR introduces. Very short statements, where
shingle Jaccard is unreliable, fall back to normalized-exact equality. Grouping is
transitive (union-find) so a problem reused three times forms one group; the
group's reported `similarity` is the weakest pairwise link, an honest lower bound.
"""

import re
from dataclasses import dataclass

from .nanonets import FIGURE_PLACEHOLDER

# Content-free LaTeX spacing/formatting commands. Non-alphanumerics are collapsed
# anyway (see `normalize`); stripping these first stops their letters -- "quad",
# "textbf" -- from surviving as spurious word tokens.
_NOISE_CMDS = re.compile(
    r"\\(?:qquad|quad|thinspace|textbf|textit|textrm|text|mathrm|mathbf|left|right|"
    r"displaystyle|dfrac|tfrac|,|;|:|!|\s)"
)
_IMG_TAG = re.compile(r"<img[^>]*>", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _strip_markdown_images(text: str) -> str:
    """Remove inline Markdown images, including paths with parentheses.

    A regex that stops at the first ``)`` leaks the tail of destinations such
    as ``![](figures/(draft)/problem.png)`` into duplicate matching.  Walk the
    brackets and parentheses instead so the entire image reference is ignored.
    """
    pieces = []
    cursor = 0
    search_from = 0

    while True:
        start = text.find("![", search_from)
        if start < 0:
            pieces.append(text[cursor:])
            return "".join(pieces)

        bracket_depth = 1
        pos = start + 2
        while pos < len(text) and bracket_depth:
            if text[pos] == "\\":
                pos += 2
                continue
            if text[pos] == "[":
                bracket_depth += 1
            elif text[pos] == "]":
                bracket_depth -= 1
            pos += 1

        if bracket_depth or pos >= len(text) or text[pos] != "(":
            search_from = start + 2
            continue

        paren_depth = 1
        pos += 1
        while pos < len(text) and paren_depth:
            if text[pos] == "\\":
                pos += 2
                continue
            if text[pos] == "(":
                paren_depth += 1
            elif text[pos] == ")":
                paren_depth -= 1
            pos += 1

        if paren_depth:
            search_from = start + 2
            continue

        pieces.append(text[cursor:start])
        pieces.append(" ")
        cursor = pos
        search_from = pos


def normalize(text: str) -> str:
    """Reduce a statement to a comparable alphanumeric word stream.

    Drops figure placeholders and image refs, lowercases, removes content-free
    LaTeX commands, then collapses every run of non-alphanumerics to one space.
    What remains -- the problem's words and numbers -- is robust to OCR
    punctuation/spacing drift while keeping enough signal to keep distinct
    problems apart.
    """
    t = text.replace(FIGURE_PLACEHOLDER, " ")
    t = _strip_markdown_images(t)
    t = _IMG_TAG.sub(" ", t)
    t = t.lower()
    t = _NOISE_CMDS.sub(" ", t)
    t = _NON_ALNUM.sub(" ", t)
    return t.strip()


def shingles(text: str, k: int) -> frozenset:
    """The set of length-`k` character substrings of `text`."""
    if len(text) <= k:
        return frozenset({text}) if text else frozenset()
    return frozenset(text[i : i + k] for i in range(len(text) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity of two sets: |a ∩ b| / |a ∪ b| (1.0 if both empty)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter) if inter else 0.0


@dataclass(frozen=True)
class ProblemRef:
    """A problem's location: which test, and its problem-number key."""

    test: str
    problem: str


@dataclass
class DuplicateGroup:
    """A set of mutually-duplicate problems sharing one comparison scope.

    `similarity` is the minimum pairwise similarity among members (the weakest
    link), so it is a conservative floor on how alike the group is.
    """

    scope: str
    members: list  # list[ProblemRef], sorted by (test, problem)
    similarity: float


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def components(self):
        groups: dict = {}
        for x in self.parent:
            groups.setdefault(self.find(x), []).append(x)
        return list(groups.values())


def _problem_sort_key(problem: str):
    """Sort problem keys numerically when possible ("2" before "10")."""
    return (0, int(problem)) if problem.lstrip("-").isdigit() else (1, problem)


def _similarity(a: dict, b: dict, min_shingle_len: int) -> float:
    """Similarity of two prepared entries.

    Below `min_shingle_len` normalized characters, shingle Jaccard is noisy (a
    terse prompt overlaps many others), so short statements are compared by
    normalized-exact equality instead.
    """
    na, nb = a["norm"], b["norm"]
    if len(na) < min_shingle_len or len(nb) < min_shingle_len:
        return 1.0 if na and na == nb else 0.0
    return jaccard(a["sh"], b["sh"])


def find_duplicate_groups(entries, threshold, k, min_shingle_len):
    """Group near-duplicate problems.

    `entries` is an iterable of ``(scope, ProblemRef, statement_text)``. Only
    problems sharing a `scope` are compared. Returns `DuplicateGroup`s (each with
    >=2 members), sorted by scope then first member.
    """
    prepared = []
    for scope, ref, text in entries:
        norm = normalize(text)
        prepared.append(
            {"scope": scope, "ref": ref, "norm": norm, "sh": shingles(norm, k)}
        )

    by_scope: dict = {}
    for i, e in enumerate(prepared):
        by_scope.setdefault(e["scope"], []).append(i)

    groups = []
    for scope in sorted(by_scope):
        idxs = by_scope[scope]
        uf = _UnionFind(idxs)
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                if _similarity(prepared[ia], prepared[ib], min_shingle_len) >= threshold:
                    uf.union(ia, ib)
        for comp in uf.components():
            if len(comp) < 2:
                continue
            pair_sims = [
                _similarity(prepared[x], prepared[y], min_shingle_len)
                for pos, x in enumerate(comp)
                for y in comp[pos + 1 :]
            ]
            members = sorted(
                (prepared[i]["ref"] for i in comp),
                key=lambda r: (r.test, _problem_sort_key(r.problem)),
            )
            groups.append(
                DuplicateGroup(
                    scope=scope,
                    members=members,
                    similarity=round(min(pair_sims), 4),
                )
            )

    groups.sort(
        key=lambda g: (g.scope, g.members[0].test, _problem_sort_key(g.members[0].problem))
    )
    return groups
