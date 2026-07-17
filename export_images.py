#!/usr/bin/env python3
"""Export every image out of the parse output tree into ~/MathWiki/images/,
preserving the relative folder structure. Copies only image files; nothing else.

    python export_images.py                 # out/ -> ~/MathWiki/images/
    python export_images.py out m0          # multiple source roots
    python export_images.py --dest ~/foo    # custom destination
    python export_images.py --move          # move instead of copy
    python export_images.py --dry-run       # show what would happen
    python export_images.py --prune         # after export, delete dest images
                                            #   no longer produced by the source
"""
import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}


def _images(root):
    return (p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def export(sources, dest, move=False, dry_run=False, prune=False):
    dest = dest.expanduser()
    copied = skipped = 0
    expected = set()  # dest paths the current sources map to
    for src in sources:
        src = src.expanduser().resolve()
        if not src.is_dir():
            print(f"! skipping missing source: {src}")
            continue
        for path in sorted(_images(src)):
            # Mirror the source's internal structure directly under dest.
            target = dest / path.relative_to(src)
            expected.add(target)
            if target.exists() and not move:
                skipped += 1
                continue
            if dry_run:
                print(f"{'MOVE' if move else 'COPY'} {path} -> {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                (shutil.move if move else shutil.copy2)(str(path), str(target))
            copied += 1
    verb = "would " if dry_run else ""
    print(f"\n{verb}{'moved' if move else 'copied'} {copied} images "
          f"({skipped} already present, skipped) -> {dest}")

    # Orphans: images already in dest that the current sources no longer produce.
    orphans = sorted(p for p in _images(dest) if p not in expected) if dest.is_dir() else []
    if orphans:
        print(f"\n{len(orphans)} image(s) in {dest} are NOT in the current source "
              f"(dropped/renamed since a previous export):")
        for p in orphans:
            print(f"  {p.relative_to(dest)}")
        if prune:
            for p in orphans:
                if dry_run:
                    print(f"DELETE {p}")
                else:
                    p.unlink()
            print(f"\n{verb}deleted {len(orphans)} orphan(s).")
        else:
            print("\n(re-run with --prune to delete these; --prune --dry-run to preview)")
    else:
        print("\nNo orphans: every image in dest is still produced by the source.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", default=["out"],
                    help="source directory roots (default: out)")
    ap.add_argument("--dest", default="~/MathWiki/images",
                    help="destination root (default: ~/MathWiki/images)")
    ap.add_argument("--move", action="store_true",
                    help="move files instead of copying")
    ap.add_argument("--dry-run", action="store_true",
                    help="print actions without touching the filesystem")
    ap.add_argument("--prune", action="store_true",
                    help="delete dest images no longer produced by the source")
    args = ap.parse_args()
    export([Path(s) for s in args.sources], Path(args.dest),
           move=args.move, dry_run=args.dry_run, prune=args.prune)
