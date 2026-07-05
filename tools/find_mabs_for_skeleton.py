#!/usr/bin/env python3
r"""Find FC2 MAB files whose routing masks match a given .skeleton.

This is meant for hash-named ``.unknown`` dumps.  Give it a Ubisoft LKS
``.skeleton`` file and a root folder; it recursively detects likely FC2 MABs
and reports the clips whose MAB masks validate against that skeleton's bone
count/order.

Examples:

  python tools/find_mabs_for_skeleton.py C:\Temp\FC2_Extracted\graphics\weapons\primary\ak47\ak47_ref.skeleton C:\Temp\FC2_Extracted --unknown-only
  python tools/find_mabs_for_skeleton.py C:\Temp\FC2_Extracted\graphics\weapons\primary\ak47\ak47_ref.skeleton C:\Temp\FC2_Extracted --unknown-only --csv C:\Temp\ak47_mab_matches.csv --copy-to C:\Temp\AK47_MABs

By default the script reports exact skeleton-domain matches.  Use
``--scan-offsets`` only for research into combined rigs, because small
skeletons can produce noisy partial matches inside a larger character domain.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from scan_mab_files import (
        FC2_SECTION_LABELS,
        FC2_SECTION_TABLE_OFF,
        SKIP,
        detect_fc2_mab,
        iter_files,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scan_mab_files import (  # type: ignore
        FC2_SECTION_LABELS,
        FC2_SECTION_TABLE_OFF,
        SKIP,
        detect_fc2_mab,
        iter_files,
    )


MASK_CONST_OFF = 0x10
MASK_ANIM_OFF = 0x24
MASK_TRANS_OFF = 0x38
MASK_SLOT = 0x14
MAX_MASK_BITS = MASK_SLOT * 8


@dataclass
class SkeletonMatch:
    path: str
    size: int
    match_type: str
    confidence: int
    anim_length: float
    frame_count: int
    rotation_tracks: int
    constant_rotations: int
    translation_tracks: int
    skeleton_bones: int
    domain_size: int
    domain_offset: int
    block_anim_bits: int
    block_const_bits: int
    block_trans_bits: int
    animated_bones: str
    constant_bones: str
    translated_bones: str
    string_hits: str


def _read_u16(data: bytes, off: int) -> int | None:
    if off < 0 or off + 2 > len(data):
        return None
    return struct.unpack_from("<H", data, off)[0]


def _read_i32(data: bytes, off: int) -> int | None:
    if off < 0 or off + 4 > len(data):
        return None
    return struct.unpack_from("<i", data, off)[0]


def _bits(data: bytes, off: int, count: int) -> list[int]:
    return [(data[off + i // 8] >> (i % 8)) & 1 for i in range(count)]


def _pop(bits: list[int]) -> int:
    return sum(bits)


def _safe_relpath(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except Exception:
        return Path(path.name)


def read_lks_bone_names(skeleton_path: Path) -> list[str]:
    """Minimal FC2 LKS parser: return bone names in skeleton order."""
    data = skeleton_path.read_bytes()
    if len(data) < 80 or data[:3] != b"LKS":
        raise ValueError(f"not an LKS skeleton: {skeleton_path}")
    bone_count = struct.unpack_from("<H", data, 16)[0]
    root_name_len = struct.unpack_from("<I", data, 74)[0]
    if not (0 < root_name_len < 256):
        raise ValueError(f"bad root name length in skeleton: {skeleton_path}")
    names = [data[78:78 + root_name_len].decode("latin-1")]
    pos = 78 + root_name_len + 1

    def name_at(block_pos: int, k: int) -> tuple[str, int] | None:
        if block_pos + k + 4 > len(data):
            return None
        name_len = struct.unpack_from("<I", data, block_pos + k)[0]
        name_pos = block_pos + k + 4
        if not (0 < name_len < 256) or name_pos + name_len + 1 > len(data):
            return None
        raw = data[name_pos:name_pos + name_len]
        if not all(0x20 <= c < 0x7F for c in raw):
            return None
        if data[name_pos + name_len] not in (0, 1):
            return None
        return raw.decode("latin-1"), name_pos + name_len + 1

    for _ in range(1, bone_count):
        hit = None
        for k in (59, 67, 71, 75):
            hit = name_at(pos, k)
            if hit:
                break
        if hit is None:
            raise ValueError(f"could not parse bone {len(names)} in {skeleton_path}")
        names.append(hit[0])
        pos = hit[1]
    return names


def parse_raw_offsets(data: bytes) -> tuple[int, ...] | None:
    if len(data) < FC2_SECTION_TABLE_OFF + 9 * 4:
        return None
    try:
        return struct.unpack_from("<9i", data, FC2_SECTION_TABLE_OFF)
    except struct.error:
        return None


def section_sizes(raw_offsets: tuple[int, ...], file_size: int) -> dict[str, int]:
    present = sorted((off, FC2_SECTION_LABELS[i])
                     for i, off in enumerate(raw_offsets) if off > 0)
    sizes: dict[str, int] = {}
    filesize_no_header = file_size - SKIP
    for idx, (off, label) in enumerate(present):
        next_off = present[idx + 1][0] if idx + 1 < len(present) else filesize_no_header
        sizes[label] = max(0, next_off - off)
    return sizes


def translation_track_count(data: bytes, raw_offsets: tuple[int, ...],
                            sizes: dict[str, int]) -> int:
    offsets_raw = raw_offsets[5]
    if offsets_raw > 0:
        off = offsets_raw + SKIP
        if sizes.get("Offsets", 0) >= 8:
            tc = _read_u16(data, off)
            if tc is not None and tc > 0:
                return tc
    unk_raw = raw_offsets[4]
    if unk_raw > 0:
        count = _read_i32(data, unk_raw + SKIP)
        if count is not None and 0 <= count < 1024:
            return count
    return 0


def selected_names(names: list[str], bits: list[int], offset: int = 0) -> list[str]:
    out = []
    for i, name in enumerate(names):
        bi = offset + i
        if 0 <= bi < len(bits) and bits[bi]:
            out.append(name)
    return out


def validate_domain(anim_bits: list[int], const_bits: list[int],
                    rot_tracks: int, const_count: int, domain_size: int) -> bool:
    if domain_size <= 0 or domain_size > len(anim_bits):
        return False
    a = anim_bits[:domain_size]
    c = const_bits[:domain_size]
    return (
        _pop(a) == rot_tracks
        and _pop(c) == const_count
        and not any(x and y for x, y in zip(a, c))
    )


def find_string_hits(data: bytes, names: list[str]) -> list[str]:
    lower = data.lower()
    hits = []
    for name in names:
        raw = name.encode("latin-1", "ignore")
        if len(raw) >= 3 and raw.lower() in lower:
            hits.append(name)
    return hits


def inspect_file(path: Path, skeleton_names: list[str], min_confidence: int,
                 scan_offsets: bool, min_block_bits: int,
                 include_translation_only: bool, probe_size: int) -> list[SkeletonMatch]:
    size = path.stat().st_size
    if size < FC2_SECTION_TABLE_OFF + 9 * 4:
        return []
    with path.open("rb") as f:
        data = f.read(min(size, probe_size))
    mab = detect_fc2_mab(path, data, size, min_confidence)
    if mab is None:
        return []

    raw_offsets = parse_raw_offsets(data)
    if raw_offsets is None:
        return []
    sizes = section_sizes(raw_offsets, size)
    trans_tracks = translation_track_count(data, raw_offsets, sizes)
    n_bones = len(skeleton_names)

    anim_bits = _bits(data, MASK_ANIM_OFF, MAX_MASK_BITS)
    const_bits = _bits(data, MASK_CONST_OFF, MAX_MASK_BITS)
    trans_bits = _bits(data, MASK_TRANS_OFF, MAX_MASK_BITS)

    matches: list[SkeletonMatch] = []
    string_hits = find_string_hits(data, skeleton_names)

    exact_rot = validate_domain(
        anim_bits, const_bits, mab.rotation_tracks, mab.constant_rotations, n_bones
    )

    print("")
    print(mab.rotation_tracks)
    print(mab.constant_rotations)
    print(mab.anim_length)
    print(mab.frame_count)
    print("")

    exact_trans = trans_tracks > 0 and _pop(trans_bits[:n_bones]) == trans_tracks
    if exact_rot or (include_translation_only and exact_trans):
        matches.append(SkeletonMatch(
            path=str(path),
            size=size,
            match_type="exact" if exact_rot else "translation-only",
            confidence=mab.confidence,
            anim_length=mab.anim_length,
            frame_count=mab.frame_count,
            rotation_tracks=mab.rotation_tracks,
            constant_rotations=mab.constant_rotations,
            translation_tracks=trans_tracks,
            skeleton_bones=n_bones,
            domain_size=n_bones,
            domain_offset=0,
            block_anim_bits=_pop(anim_bits[:n_bones]),
            block_const_bits=_pop(const_bits[:n_bones]),
            block_trans_bits=_pop(trans_bits[:n_bones]),
            animated_bones=", ".join(selected_names(skeleton_names, anim_bits)),
            constant_bones=", ".join(selected_names(skeleton_names, const_bits)),
            translated_bones=", ".join(selected_names(skeleton_names, trans_bits)),
            string_hits=", ".join(string_hits),
        ))

    if scan_offsets:
        for domain_size in range(n_bones + 1, MAX_MASK_BITS + 1):
            if not validate_domain(
                anim_bits, const_bits, mab.rotation_tracks,
                mab.constant_rotations, domain_size
            ):
                continue
            if trans_tracks > 0 and _pop(trans_bits[:domain_size]) != trans_tracks:
                continue
            for offset in range(0, domain_size - n_bones + 1):
                block_anim = _pop(anim_bits[offset:offset + n_bones])
                block_const = _pop(const_bits[offset:offset + n_bones])
                block_trans = _pop(trans_bits[offset:offset + n_bones])
                if block_anim + block_const + block_trans < min_block_bits:
                    continue
                matches.append(SkeletonMatch(
                    path=str(path),
                    size=size,
                    match_type="offset-candidate",
                    confidence=mab.confidence,
                    anim_length=mab.anim_length,
                    frame_count=mab.frame_count,
                    rotation_tracks=mab.rotation_tracks,
                    constant_rotations=mab.constant_rotations,
                    translation_tracks=trans_tracks,
                    skeleton_bones=n_bones,
                    domain_size=domain_size,
                    domain_offset=offset,
                    block_anim_bits=block_anim,
                    block_const_bits=block_const,
                    block_trans_bits=block_trans,
                    animated_bones=", ".join(selected_names(skeleton_names, anim_bits, offset)),
                    constant_bones=", ".join(selected_names(skeleton_names, const_bits, offset)),
                    translated_bones=", ".join(selected_names(skeleton_names, trans_bits, offset)),
                    string_hits=", ".join(string_hits),
                ))
            break

    return matches


def copy_match(match: SkeletonMatch, root: Path, copy_to: Path) -> Path:
    src = Path(match.path)
    rel = _safe_relpath(src, root)
    dst = (copy_to / rel).with_suffix(".mab")
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


def write_csv(path: Path, matches: list[SkeletonMatch]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(SkeletonMatch.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for match in matches:
            writer.writerow(asdict(match))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Find FC2 MAB files matching an input .skeleton routing domain."
    )
    ap.add_argument("skeleton", type=Path, help="Input FC2 .skeleton/LKS file.")
    ap.add_argument("root", type=Path, help="Folder to scan recursively.")
    ap.add_argument("--unknown-only", action="store_true",
                    help="Only scan files ending in .unknown.")
    ap.add_argument("--min-confidence", type=int, default=14,
                    help="Minimum MAB detector confidence. Default: 14.")
    ap.add_argument("--probe-size", type=int, default=1024 * 1024,
                    help="Maximum bytes read per file. Default: 1 MiB.")
    ap.add_argument("--scan-offsets", action="store_true",
                    help="Also report noisy combined-rig offset candidates.")
    ap.add_argument("--include-translation-only", action="store_true",
                    help="Also report clips whose translation mask alone matches. Noisy for small skeletons.")
    ap.add_argument("--min-block-bits", type=int, default=1,
                    help="Minimum bits inside skeleton block for --scan-offsets. Default: 1.")
    ap.add_argument("--csv", type=Path, help="Write match list to CSV.")
    ap.add_argument("--json", type=Path, help="Write match list to JSON.")
    ap.add_argument("--copy-to", type=Path,
                    help="Copy matched files into this folder with a .mab extension.")
    args = ap.parse_args(argv)

    if not args.skeleton.exists():
        print(f"error: skeleton does not exist: {args.skeleton}", file=sys.stderr)
        return 2
    if not args.root.exists() or not args.root.is_dir():
        print(f"error: root folder does not exist: {args.root}", file=sys.stderr)
        return 2

    skeleton_names = read_lks_bone_names(args.skeleton)
    print(f"Skeleton: {args.skeleton}")
    print(f"Bones: {len(skeleton_names)}")
    print("Bone order: " + ", ".join(skeleton_names))

    matches: list[SkeletonMatch] = []
    scanned = 0
    for path in iter_files(args.root, args.unknown_only):
        scanned += 1
        try:
            matches.extend(inspect_file(
                path, skeleton_names, args.min_confidence,
                args.scan_offsets, args.min_block_bits,
                args.include_translation_only, args.probe_size
            ))
        except OSError:
            continue

    matches.sort(key=lambda m: (
        0 if m.match_type == "exact" else 1 if m.match_type == "translation-only" else 2,
        -m.block_anim_bits,
        -m.block_trans_bits,
        m.path.lower(),
    ))

    print(f"Scanned files: {scanned}")
    print(f"Matching MABs: {len(matches)}")
    for match in matches:
        bones = match.animated_bones or match.translated_bones or match.constant_bones
        if len(bones) > 120:
            bones = bones[:117] + "..."
        print(
            f"[{match.match_type}] {match.path} "
            f"frames={match.frame_count} rot={match.rotation_tracks} "
            f"const={match.constant_rotations} trans={match.translation_tracks} "
            f"domain={match.domain_size}@{match.domain_offset} "
            f"block(rot/const/trans)="
            f"{match.block_anim_bits}/{match.block_const_bits}/{match.block_trans_bits} "
            f"bones={bones}"
        )

    if args.csv:
        write_csv(args.csv, matches)
        print(f"CSV written: {args.csv}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([asdict(m) for m in matches], indent=2),
            encoding="utf-8",
        )
        print(f"JSON written: {args.json}")

    if args.copy_to:
        seen: set[str] = set()
        copied = 0
        for match in matches:
            if match.path in seen:
                continue
            seen.add(match.path)
            copy_match(match, args.root, args.copy_to)
            copied += 1
        print(f"Copied {copied} unique file(s) to: {args.copy_to}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
