#!/usr/bin/env python3
r"""Recursively scan extracted Dunia files for likely MAB animations.

Far Cry 2 MAB files do not use a plain ASCII magic at byte 0.  Many extracted
resources can therefore end up as hash-named ``.unknown`` files.  This scanner
uses the FC2 MAB header layout and section table to find renamed files.

Examples:

  python tools/scan_mab_files.py C:\Temp\FC2_Extracted --unknown-only
  python tools/scan_mab_files.py C:\Temp\FC2_Extracted --csv mab_hits.csv
  python tools/scan_mab_files.py C:\Temp\FC2_Extracted --copy-to C:\Temp\MAB_Hits
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SKIP = 16
FC2_VERSION = 0x4C
FC2_ANIMLEN_OFF = 0x84
FC2_SECTION_TABLE_OFF = 0x88
FC2_SECTION_LABELS = (
    "UnkSec2",
    "UnkSec1",
    "ConstantRot",
    "Keyframes",
    "UnkSec3",
    "Offsets",
    "Events",
    "UnkSec4",
    "UnkSec5",
)


@dataclass
class MabHit:
    path: str
    size: int
    confidence: int
    reason: str
    version: str
    anim_length: float
    frame_count: int
    rotation_tracks: int
    constant_rotations: int
    translation_mask_bits: int
    section_offsets: str
    section_sizes: str


def _popcount_bytes(data: bytes) -> int:
    return sum(int(b).bit_count() for b in data)


def _read_i32(data: bytes, off: int) -> int | None:
    if off < 0 or off + 4 > len(data):
        return None
    return struct.unpack_from("<i", data, off)[0]


def _read_u16(data: bytes, off: int) -> int | None:
    if off < 0 or off + 2 > len(data):
        return None
    return struct.unpack_from("<H", data, off)[0]


def _read_u32(data: bytes, off: int) -> int | None:
    if off < 0 or off + 4 > len(data):
        return None
    return struct.unpack_from("<I", data, off)[0]


def _safe_relpath(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except Exception:
        return Path(path.name)


def _section_sizes(raw_offsets: tuple[int, ...], file_size: int) -> dict[str, int]:
    present = sorted((off, FC2_SECTION_LABELS[i])
                     for i, off in enumerate(raw_offsets) if off > 0)
    sizes: dict[str, int] = {}
    filesize_no_header = file_size - SKIP
    for idx, (off, label) in enumerate(present):
        next_off = present[idx + 1][0] if idx + 1 < len(present) else filesize_no_header
        sizes[label] = max(0, next_off - off)
    return sizes


def detect_fc2_mab(path: Path, probe: bytes, file_size: int,
                   min_confidence: int) -> MabHit | None:
    """Return a MabHit if the file structurally looks like an FC2 MAB."""
    if file_size < FC2_SECTION_TABLE_OFF + 9 * 4:
        return None

    score = 0
    reasons: list[str] = []

    if probe[0] != FC2_VERSION:
        return None
    score += 2
    reasons.append("version=0x4C")

    # Observed FC2 MAB header bytes: 4C 00 03 00 ...
    if len(probe) >= 4 and probe[1:4] == b"\x00\x03\x00":
        score += 2
        reasons.append("fc2-header")

    # Strong marker observed immediately before anim length in validated FC2 MABs.
    if len(probe) >= FC2_ANIMLEN_OFF and probe[FC2_ANIMLEN_OFF - 4:FC2_ANIMLEN_OFF] == b"AnD\x1a":
        score += 5
        reasons.append("AnD-marker")

    try:
        anim_length = struct.unpack_from("<f", probe, FC2_ANIMLEN_OFF)[0]
    except struct.error:
        return None
    if math.isfinite(anim_length) and 0.0 <= anim_length <= 600.0:
        score += 2
        reasons.append("sane-animlen")
    else:
        return None

    try:
        raw_offsets = struct.unpack_from("<9i", probe, FC2_SECTION_TABLE_OFF)
    except struct.error:
        return None

    positive = [o for o in raw_offsets if o > 0]
    if len(positive) < 3:
        return None
    if len(set(positive)) != len(positive):
        return None
    if any(o + SKIP >= file_size for o in positive):
        return None
    if any(o < 0 for o in raw_offsets):
        return None

    score += 3
    reasons.append("section-table")

    if all((o % 16) == 0 for o in positive):
        score += 2
        reasons.append("aligned-sections")

    sizes = _section_sizes(raw_offsets, file_size)
    if sizes.get("Keyframes", 0) > 0:
        score += 2
        reasons.append("keyframes-section")
    else:
        return None

    key_abs = raw_offsets[3] + SKIP if raw_offsets[3] > 0 else 0
    rot_tracks = _read_u16(probe, key_abs)
    frame_count = _read_u16(probe, key_abs + 2)
    sample_rate = _read_u32(probe, key_abs + 4)
    if rot_tracks is None or frame_count is None or sample_rate is None:
        return None
    if 0 <= rot_tracks < 1024 and 0 < frame_count < 100000 and sample_rate < 100000:
        score += 3
        reasons.append("keyframe-header")
    else:
        return None

    const_count = -1
    const_abs = raw_offsets[2] + SKIP if raw_offsets[2] > 0 else 0
    if const_abs:
        c = _read_i32(probe, const_abs)
        if c is not None and 0 <= c < 1024:
            const_count = c
            const_size = sizes.get("ConstantRot", 0)
            if const_size == 0 or 8 + c * 6 <= const_size:
                score += 3
                reasons.append("constant-rot-header")

    if len(probe) >= 0x4C:
        const_bits = _popcount_bytes(probe[0x10:0x24])
        anim_bits = _popcount_bytes(probe[0x24:0x38])
        trans_bits = _popcount_bytes(probe[0x38:0x4C])
        if anim_bits == rot_tracks:
            score += 3
            reasons.append("anim-mask-popcount")
        if const_count >= 0 and const_bits == const_count:
            score += 3
            reasons.append("const-mask-popcount")
    else:
        trans_bits = 0

    if score < min_confidence:
        return None

    section_offsets = ";".join(
        f"{label}=0x{raw_offsets[i] + SKIP:X}" if raw_offsets[i] > 0 else f"{label}=0"
        for i, label in enumerate(FC2_SECTION_LABELS)
    )
    section_sizes = ";".join(
        f"{label}=0x{sizes.get(label, 0):X}" for label in FC2_SECTION_LABELS
    )

    return MabHit(
        path=str(path),
        size=file_size,
        confidence=score,
        reason=",".join(reasons),
        version="Far Cry 2",
        anim_length=anim_length,
        frame_count=frame_count,
        rotation_tracks=rot_tracks,
        constant_rotations=const_count,
        translation_mask_bits=trans_bits,
        section_offsets=section_offsets,
        section_sizes=section_sizes,
    )


def iter_files(root: Path, unknown_only: bool) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
        for name in filenames:
            path = Path(dirpath) / name
            if unknown_only and path.suffix.lower() != ".unknown":
                continue
            yield path


def copy_hit(hit: MabHit, root: Path, copy_to: Path) -> Path:
    src = Path(hit.path)
    rel = _safe_relpath(src, root)
    dst = copy_to / rel
    dst = dst.with_suffix(".mab")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        stem = dst.stem
        suffix = dst.suffix
        for i in range(1, 10000):
            candidate = dst.with_name(f"{stem}_{i:03d}{suffix}")
            if not candidate.exists():
                dst = candidate
                break
    shutil.copy2(src, dst)
    return dst


def scan(root: Path, unknown_only: bool, min_confidence: int,
         probe_size: int) -> list[MabHit]:
    hits: list[MabHit] = []
    for path in iter_files(root, unknown_only):
        try:
            size = path.stat().st_size
            if size < FC2_SECTION_TABLE_OFF + 9 * 4:
                continue
            with path.open("rb") as f:
                probe = f.read(min(size, probe_size))
        except OSError:
            continue
        hit = detect_fc2_mab(path, probe, size, min_confidence)
        if hit is not None:
            hits.append(hit)
    hits.sort(key=lambda h: (-h.confidence, h.path.lower()))
    return hits


def write_csv(path: Path, hits: list[MabHit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(hits[0]).keys()) if hits else [
            "path", "size", "confidence", "reason", "version", "anim_length",
            "frame_count", "rotation_tracks", "constant_rotations",
            "translation_mask_bits", "section_offsets", "section_sizes",
        ])
        writer.writeheader()
        for hit in hits:
            writer.writerow(asdict(hit))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Recursively detect hash-named or extensionless Far Cry 2 MAB files."
    )
    ap.add_argument("root", help="Folder to scan recursively.")
    ap.add_argument("--unknown-only", action="store_true",
                    help="Only scan files ending in .unknown.")
    ap.add_argument("--min-confidence", type=int, default=14,
                    help="Minimum detector confidence score. Default: 14.")
    ap.add_argument("--probe-size", type=int, default=1024 * 1024,
                    help="Maximum bytes read per file. Default: 1 MiB.")
    ap.add_argument("--csv", type=Path,
                    help="Write full hit list to a CSV file.")
    ap.add_argument("--json", type=Path,
                    help="Write full hit list to a JSON file.")
    ap.add_argument("--copy-to", type=Path,
                    help="Copy detected files into this folder with a .mab extension.")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(f"error: root folder does not exist: {root}", file=sys.stderr)
        return 2

    hits = scan(root, args.unknown_only, args.min_confidence, args.probe_size)

    print(f"Scanned: {root}")
    print(f"Likely FC2 MAB files: {len(hits)}")
    for hit in hits:
        print(
            f"[{hit.confidence:02d}] {hit.path} "
            f"frames={hit.frame_count} rot={hit.rotation_tracks} "
            f"const={hit.constant_rotations} transMask={hit.translation_mask_bits} "
            f"len={hit.anim_length:.3f}s"
        )

    if args.csv:
        write_csv(args.csv, hits)
        print(f"CSV written: {args.csv}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([asdict(h) for h in hits], indent=2), encoding="utf-8")
        print(f"JSON written: {args.json}")

    if args.copy_to:
        copied = 0
        for hit in hits:
            copy_hit(hit, root, args.copy_to)
            copied += 1
        print(f"Copied {copied} file(s) to: {args.copy_to}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
