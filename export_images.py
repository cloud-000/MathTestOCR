#!/usr/bin/env python3
"""Export parse-tree images as transparent PNGs into ~/MathWiki/images/.

The source tree is never modified unless ``--move`` is explicitly requested.
Relative paths are preserved; non-PNG source suffixes become ``.png``.

    python3 export_images.py                 # out/ -> ~/MathWiki/images/
    python3 export_images.py out m0          # multiple source roots
    python3 export_images.py --dest ~/foo    # custom destination
    python3 export_images.py --preserve-background  # old byte-copy behavior
    python3 export_images.py --force         # rebuild existing exports
    python3 export_images.py --move          # export, then remove each source
    python3 export_images.py --dry-run       # show what would happen
    python3 export_images.py --prune         # delete destination orphans
"""

import argparse
import hashlib
import os
import shutil
import statistics
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageMath, PngImagePlugin


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
ALGORITHM_VERSION = "paper-alpha-v1"
ALGORITHM_KEY = "comp-ocr-background-removal"
SOURCE_HASH_KEY = "comp-ocr-source-sha256"
PAPER_MIN_LUMA = 200
PAPER_MAX_CHROMA = 30
ALPHA_NOISE_FLOOR = 6


class ExportCollisionError(ValueError):
    """Two source images would write the same destination path."""


def _images(root):
    return (
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _source_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _estimate_paper_color(image):
    """Return the median RGB of the brightest pale, low-chroma pixels."""
    sample = image.convert("RGB")
    sample.thumbnail((256, 256), Image.Resampling.BOX)
    candidates = []
    # Pillow 14 replaces getdata() with get_flattened_data(); retain support for
    # the older system Pillow used by this standalone script.
    pixels = (
        sample.get_flattened_data()
        if hasattr(sample, "get_flattened_data")
        else sample.getdata()
    )
    for pixel in pixels:
        lo, hi = min(pixel), max(pixel)
        luma = (299 * pixel[0] + 587 * pixel[1] + 114 * pixel[2]) // 1000
        if luma >= PAPER_MIN_LUMA and hi - lo <= PAPER_MAX_CHROMA:
            candidates.append((luma, pixel))
    if not candidates:
        return (255, 255, 255)

    # Dark strokes can touch every crop edge, so use the globally brightest
    # candidates rather than assuming the border is clean paper.
    candidates.sort(reverse=True)
    brightest = [pixel for _, pixel in candidates[:max(1, len(candidates) // 5)]]
    return tuple(
        int(statistics.median(pixel[channel] for pixel in brightest))
        for channel in range(3)
    )


def _normalized_deficit(channel, paper_value):
    """How far a channel falls below paper, normalized to an 8-bit range."""
    paper = Image.new("L", channel.size, paper_value)
    deficit = ImageChops.subtract(paper, channel)
    lut = [min(255, round(value * 255 / max(1, paper_value))) for value in range(256)]
    return deficit.point(lut)


def _unmatte_channel(channel, alpha, paper_value):
    """Remove the estimated paper contribution from one color channel."""
    return ImageMath.lambda_eval(
        lambda op: op["convert"](
            (
                op["channel"] * 255
                - paper_value * (255 - op["alpha"])
            )
            / (op["alpha"] + (op["alpha"] == 0)),
            "L",
        ),
        channel=channel,
        alpha=alpha,
    )


def remove_paper_background(image):
    """Convert a static scan crop to RGBA with pale paper made transparent.

    Alpha is derived from the strongest per-channel contrast against the
    estimated paper color. The RGB channels are then unmatted so antialiased
    strokes do not acquire pale fringes on a dark destination background.
    """
    image.seek(0)  # Static export: use only the first frame of animated inputs.
    rgba = image.convert("RGBA")
    red, green, blue, source_alpha = rgba.split()
    paper = _estimate_paper_color(rgba)

    alpha = ImageChops.lighter(
        _normalized_deficit(red, paper[0]),
        ImageChops.lighter(
            _normalized_deficit(green, paper[1]),
            _normalized_deficit(blue, paper[2]),
        ),
    )
    # Discard the faintest scanner/JPEG noise, then stretch the remaining
    # coverage back across the full alpha range.
    alpha_lut = [
        0
        if value <= ALPHA_NOISE_FLOOR
        else round((value - ALPHA_NOISE_FLOOR) * 255 / (255 - ALPHA_NOISE_FLOOR))
        for value in range(256)
    ]
    alpha = alpha.point(alpha_lut)
    alpha = ImageChops.multiply(alpha, source_alpha)

    unmatted = [
        _unmatte_channel(channel, alpha, paper_value)
        for channel, paper_value in zip((red, green, blue), paper)
    ]
    return Image.merge("RGBA", (*unmatted, alpha))


def _processed_target(dest, relative_path):
    target = dest / relative_path
    return target if target.suffix.lower() == ".png" else target.with_suffix(".png")


def _build_jobs(sources, dest, preserve_background):
    jobs = []
    targets = {}
    missing = []
    for source_root in sources:
        source_root = source_root.expanduser().resolve()
        if not source_root.is_dir():
            missing.append(source_root)
            continue
        for source in sorted(_images(source_root)):
            relative = source.relative_to(source_root)
            target = dest / relative if preserve_background else _processed_target(dest, relative)
            previous = targets.get(target)
            if previous is not None and previous != source:
                raise ExportCollisionError(
                    f"destination collision: {previous} and {source} both map to {target}"
                )
            targets[target] = source
            jobs.append((source, target))
    return jobs, set(targets), missing


def _is_current_processed(target, source_hash):
    if not target.is_file():
        return False
    try:
        with Image.open(target) as image:
            return (
                image.format == "PNG"
                and image.mode == "RGBA"
                and image.info.get(ALGORITHM_KEY) == ALGORITHM_VERSION
                and image.info.get(SOURCE_HASH_KEY) == source_hash
            )
    except (OSError, ValueError):
        return False


def _save_processed(source, target, source_hash):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with Image.open(source) as image:
            processed = remove_paper_background(image)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text(ALGORITHM_KEY, ALGORITHM_VERSION)
            metadata.add_text(SOURCE_HASH_KEY, source_hash)
            with tempfile.NamedTemporaryFile(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
            processed.save(temporary, format="PNG", pnginfo=metadata)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export(
    sources,
    dest,
    move=False,
    dry_run=False,
    prune=False,
    preserve_background=False,
    force=False,
):
    dest = dest.expanduser()
    jobs, expected, missing = _build_jobs(sources, dest, preserve_background)
    for source_root in missing:
        print(f"! skipping missing source: {source_root}")

    stats = {"processed": 0, "copied": 0, "skipped": 0, "failed": 0, "orphans": 0}
    for source, target in jobs:
        try:
            if preserve_background:
                if target.exists() and not force and not move:
                    stats["skipped"] += 1
                    continue
                if dry_run:
                    print(f"{'MOVE' if move else 'COPY'} {source} -> {target}")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    (shutil.move if move else shutil.copy2)(str(source), str(target))
                stats["copied"] += 1
                continue

            source_hash = _source_hash(source)
            if not force and not move and _is_current_processed(target, source_hash):
                stats["skipped"] += 1
                continue
            if dry_run:
                action = "PROCESS+MOVE" if move else "PROCESS"
                print(f"{action} {source} -> {target}")
            else:
                _save_processed(source, target, source_hash)
                if move:
                    source.unlink()
            stats["processed"] += 1
        except Exception as exc:  # Keep a large batch going and report every bad file.
            stats["failed"] += 1
            print(f"! failed {source}: {exc}")

    verb = "would " if dry_run else ""
    print(
        f"\n{verb}processed {stats['processed']}, copied {stats['copied']}, "
        f"skipped {stats['skipped']}, failed {stats['failed']} image(s) -> {dest}"
    )

    orphans = sorted(p for p in _images(dest) if p not in expected) if dest.is_dir() else []
    stats["orphans"] = len(orphans)
    if orphans:
        print(
            f"\n{len(orphans)} image(s) in {dest} are NOT in the current source "
            "(dropped/renamed since a previous export):"
        )
        for path in orphans:
            print(f"  {path.relative_to(dest)}")
        if prune:
            for path in orphans:
                if dry_run:
                    print(f"DELETE {path}")
                else:
                    path.unlink()
            print(f"\n{verb}deleted {len(orphans)} orphan(s).")
        else:
            print("\n(re-run with --prune to delete these; --prune --dry-run to preview)")
    else:
        print("\nNo orphans: every image in dest is still produced by the source.")
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "sources", nargs="*", default=["out"], help="source directory roots (default: out)"
    )
    parser.add_argument(
        "--dest", default="~/MathWiki/images", help="destination root (default: ~/MathWiki/images)"
    )
    parser.add_argument("--move", action="store_true", help="remove each source after successful export")
    parser.add_argument("--dry-run", action="store_true", help="print actions without touching the filesystem")
    parser.add_argument("--prune", action="store_true", help="delete destination images no longer produced")
    parser.add_argument("--force", action="store_true", help="re-export files even when the source hash matches")
    parser.add_argument(
        "--preserve-background",
        action="store_true",
        help="copy images unchanged instead of producing transparent PNGs",
    )
    args = parser.parse_args(argv)
    try:
        export(
            [Path(source) for source in args.sources],
            Path(args.dest),
            move=args.move,
            dry_run=args.dry_run,
            prune=args.prune,
            preserve_background=args.preserve_background,
            force=args.force,
        )
    except ExportCollisionError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
