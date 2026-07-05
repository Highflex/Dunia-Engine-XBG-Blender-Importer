#!/usr/bin/env python3
r"""Dump FC2 MAB bone routing for a character skeleton and optional weapon rig.

Give the tool the animation skeleton used by the main character stream, the
weapon .skeleton, and one .mab file.  The weapon skeleton is only opened when
the MAB has a structured weapon block in top-level UnkSec5.

Examples:

  python tools\dump_mab_bone_mapping_fc2.py C:\Temp\FC2_Extracted\graphics\characters\_common\pelvis_ref.skeleton C:\Temp\FC2_Extracted\graphics\weapons\primary\ak47\ak47_ref.skeleton C:\Temp\FC2_Extracted\graphics\characters\_common\animations\weapons\primary\ak47\1stge_uppb_reload_+000fw_prak4_i1.mab

  python tools/dump_mab_bone_mapping_fc2.py main.skeleton weapon.skeleton clip.mab --json clip_map.json
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


SKIP = 0x10
FC2_VERSION = 0x4C
FC2_ANIMLEN_OFF = 0x84
FC2_SECTION_TABLE_OFF = 0x88
MASK_SLOT = 0x14
MAX_MASK_BITS = MASK_SLOT * 8

SECTION_LABELS = (
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

MAIN_MASKS = {
    "constant_rotation": 0x10,
    "animated_rotation": 0x24,
    "animated_translation": 0x38,
}

WEAPON_MASKS = (
    ("constant_rotation", 0x00),
    ("animated_rotation", 0x14),
    ("static_translation", 0x28),
    ("animated_translation", 0x3C),
)


@dataclass
class BoneSelection:
    count: int
    indices: list[int]
    names: list[str]


@dataclass
class MainMapping:
    skeleton_path: str
    bone_count: int
    rotation_tracks: int
    constant_rotations: int
    translation_tracks: int
    translation_source: str
    static_translation_values: int
    animated_translation_tracks: int
    masks_validate: bool
    translation_mask_validates: bool
    animated: BoneSelection
    constant: BoneSelection
    translated: BoneSelection
    notes: list[str] = field(default_factory=list)


@dataclass
class LocalSectionInfo:
    rel_offset: int
    abs_offset: int
    size: int


@dataclass
class WeaponBlockMapping:
    name: str
    slot: str
    rel_offset: int
    abs_offset: int
    size: int
    anim_length: float | None
    reference_rotation_a: tuple[float, float, float, float] | None
    reference_rotation_b: tuple[float, float, float, float] | None
    rotation_tracks: int
    frame_count: int
    sample_rate: int
    constant_rotation_count: int
    static_translation_count: int
    animated_translation_tracks: int
    animated_translation_frames: int
    animated: BoneSelection
    constant: BoneSelection
    static_translated: BoneSelection
    animated_translated: BoneSelection
    sections: dict[str, LocalSectionInfo]
    notes: list[str] = field(default_factory=list)


@dataclass
class MabMapping:
    mab_path: str
    size: int
    version: str
    anim_length: float
    frame_count: int
    sample_rate: int
    sections: dict[str, LocalSectionInfo]
    main: MainMapping
    weapon_skeleton_path: str | None
    weapon_blocks: list[WeaponBlockMapping]


def _read_u16(data: bytes, off: int) -> int | None:
    if off < 0 or off + 2 > len(data):
        return None
    return struct.unpack_from("<H", data, off)[0]


def _read_u32(data: bytes, off: int) -> int | None:
    if off < 0 or off + 4 > len(data):
        return None
    return struct.unpack_from("<I", data, off)[0]


def _read_i32(data: bytes, off: int) -> int | None:
    if off < 0 or off + 4 > len(data):
        return None
    return struct.unpack_from("<i", data, off)[0]


def _read_f32(data: bytes, off: int) -> float | None:
    if off < 0 or off + 4 > len(data):
        return None
    return struct.unpack_from("<f", data, off)[0]


def _read_float4(data: bytes, off: int) -> tuple[float, float, float, float] | None:
    if off < 0 or off + 16 > len(data):
        return None
    return struct.unpack_from("<4f", data, off)


def _bits(data: bytes, off: int, count: int) -> list[int]:
    return [(data[off + i // 8] >> (i % 8)) & 1 for i in range(count)]


def _pop(bits: list[int]) -> int:
    return sum(bits)


def _selection(names: list[str], bits: list[int]) -> BoneSelection:
    indices = [i for i, _name in enumerate(names) if i < len(bits) and bits[i]]
    picked = [names[i] for i in indices]
    return BoneSelection(count=len(picked), indices=indices, names=picked)


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


def parse_sections(data: bytes) -> tuple[tuple[int, ...], dict[str, LocalSectionInfo]]:
    if len(data) < FC2_SECTION_TABLE_OFF + 9 * 4:
        raise ValueError("file is too small for an FC2 MAB section table")
    raw = struct.unpack_from("<9i", data, FC2_SECTION_TABLE_OFF)
    present = sorted((off, SECTION_LABELS[i]) for i, off in enumerate(raw) if off > 0)
    file_size_no_header = len(data) - SKIP
    sections: dict[str, LocalSectionInfo] = {}
    for idx, (off, label) in enumerate(present):
        next_off = present[idx + 1][0] if idx + 1 < len(present) else file_size_no_header
        sections[label] = LocalSectionInfo(
            rel_offset=off,
            abs_offset=off + SKIP,
            size=max(0, next_off - off),
        )
    return raw, sections


def _section_header(data: bytes, sections: dict[str, LocalSectionInfo],
                    label: str) -> tuple[int, int, int]:
    sec = sections.get(label)
    if sec is None or sec.size < 8:
        return 0, 0, 0
    return (
        _read_u16(data, sec.abs_offset) or 0,
        _read_u16(data, sec.abs_offset + 2) or 0,
        _read_u32(data, sec.abs_offset + 4) or 0,
    )


def main_static_translation_count(data: bytes, sections: dict[str, LocalSectionInfo]) -> int:
    sec = sections.get("UnkSec3")
    if sec is not None and sec.size >= 8:
        count = _read_i32(data, sec.abs_offset)
        if count is not None and 0 <= count < 4096:
            return count
    return 0


def main_translation_track_info(data: bytes, sections: dict[str, LocalSectionInfo]) -> tuple[int, str, int, int]:
    animated_tracks, _off_frames, _off_rate = _section_header(data, sections, "Offsets")
    static_values = main_static_translation_count(data, sections)
    if animated_tracks and static_values:
        return (
            animated_tracks,
            "Offsets animated vec3 tracks + UnkSec3 static vec3 values",
            static_values,
            animated_tracks,
        )
    if animated_tracks:
        return animated_tracks, "Offsets animated vec3 tracks", static_values, animated_tracks
    if static_values:
        return static_values, "UnkSec3 static vec3 values", static_values, animated_tracks
    return 0, "none", static_values, animated_tracks


def build_main_mapping(data: bytes, sections: dict[str, LocalSectionInfo],
                       skeleton_path: Path, skeleton_names: list[str]) -> MainMapping:
    rotation_tracks, _frame_count, _rate = _section_header(data, sections, "Keyframes")
    constant_sec = sections.get("ConstantRot")
    constant_rotations = 0
    if constant_sec is not None and constant_sec.size >= 8:
        constant_rotations = _read_i32(data, constant_sec.abs_offset) or 0
    (translation_tracks, translation_source, static_translation_values,
     animated_translation_tracks) = main_translation_track_info(data, sections)

    n_bones = len(skeleton_names)
    const_bits = _bits(data, MAIN_MASKS["constant_rotation"], min(n_bones, MAX_MASK_BITS))
    anim_bits = _bits(data, MAIN_MASKS["animated_rotation"], min(n_bones, MAX_MASK_BITS))
    trans_bits = _bits(data, MAIN_MASKS["animated_translation"], min(n_bones, MAX_MASK_BITS))

    masks_validate = (
        _pop(anim_bits) == rotation_tracks
        and _pop(const_bits) == constant_rotations
        and not any(a and c for a, c in zip(anim_bits, const_bits))
    )
    trans_valid = (translation_tracks == 0 and _pop(trans_bits) == 0) or (
        translation_tracks > 0 and _pop(trans_bits) == translation_tracks
    )
    notes: list[str] = []
    if not masks_validate:
        notes.append(
            "rotation masks do not exactly match this skeleton "
            f"(anim bits { _pop(anim_bits) }/{rotation_tracks}, "
            f"const bits { _pop(const_bits) }/{constant_rotations})"
        )
    if not trans_valid:
        notes.append(
            "translation mask does not exactly match this skeleton "
            f"(bits { _pop(trans_bits) }/{translation_tracks})"
        )

    return MainMapping(
        skeleton_path=str(skeleton_path),
        bone_count=n_bones,
        rotation_tracks=rotation_tracks,
        constant_rotations=constant_rotations,
        translation_tracks=translation_tracks,
        translation_source=translation_source,
        static_translation_values=static_translation_values,
        animated_translation_tracks=animated_translation_tracks,
        masks_validate=masks_validate,
        translation_mask_validates=trans_valid,
        animated=_selection(skeleton_names, anim_bits),
        constant=_selection(skeleton_names, const_bits),
        translated=_selection(skeleton_names, trans_bits),
        notes=notes,
    )


def local_section_table(data: bytes, base: int, payload_size: int) -> tuple[int, ...] | None:
    if payload_size < 0xA0 or base < 0 or base + 0xA0 > len(data):
        return None
    try:
        return struct.unpack_from("<8I", data, base + 0x80)
    except struct.error:
        return None


def local_section_sizes(raw: tuple[int, ...], payload_size: int) -> dict[int, LocalSectionInfo]:
    out: dict[int, LocalSectionInfo] = {}
    positives = [off for off in raw if 0 < off < payload_size]
    for idx, off in enumerate(raw):
        if not (0 < off < payload_size):
            continue
        next_rel = min((x for x in positives if x > off), default=payload_size)
        out[idx] = LocalSectionInfo(rel_offset=off, abs_offset=0, size=max(0, next_rel - off))
    return out


def looks_like_weapon_container(data: bytes, base: int, payload_size: int) -> bool:
    raw = local_section_table(data, base, payload_size)
    if raw is None:
        return False
    # The local header is 0xA0 bytes in all observed FC2 weapon streams.
    if raw[0] < 0xA0 or raw[0] >= payload_size:
        return False
    if raw[1] and raw[1] <= raw[0]:
        return False
    positives = [off for off in raw if off > 0]
    if any(off > payload_size for off in positives):
        return False
    if sum(1 for off in positives if off < payload_size) < 3:
        return False
    return True


def _local_header(data: bytes, off: int) -> tuple[int, int, int]:
    return (
        _read_u16(data, off) or 0,
        _read_u16(data, off + 2) or 0,
        _read_u32(data, off + 4) or 0,
    )


def parse_weapon_block(data: bytes, weapon_names: list[str], base: int, payload_size: int,
                       name: str, slot: str, rel_offset: int) -> WeaponBlockMapping:
    raw = local_section_table(data, base, payload_size)
    if raw is None:
        raise ValueError(f"{name}: not enough data for local section table")

    local_sections = local_section_sizes(raw, payload_size)
    sections_by_name: dict[str, LocalSectionInfo] = {}
    section_labels = {
        0: "ConstantRot",
        1: "Keyframes",
        2: "StaticTranslationValues",
        3: "AnimatedTranslationTracks",
        4: "Unknown4",
        5: "Unknown5",
        6: "SecondaryOrTail6",
        7: "SecondaryOrTail7",
    }
    for idx, info in local_sections.items():
        sections_by_name[section_labels.get(idx, f"Local{idx}")] = LocalSectionInfo(
            rel_offset=info.rel_offset,
            abs_offset=base + info.rel_offset,
            size=info.size,
        )

    n_bones = min(len(weapon_names), MAX_MASK_BITS)
    mask_bits = {
        label: _bits(data, base + off, n_bones)
        for label, off in WEAPON_MASKS
    }

    const_count = 0
    const_info = local_sections.get(0)
    if const_info is not None and const_info.size >= 8:
        const_count = _read_i32(data, base + const_info.rel_offset) or 0

    rot_tracks = frame_count = sample_rate = 0
    key_info = local_sections.get(1)
    if key_info is not None and key_info.size >= 8:
        rot_tracks, frame_count, sample_rate = _local_header(data, base + key_info.rel_offset)

    static_count = 0
    static_info = local_sections.get(2)
    if static_info is not None and static_info.size >= 8:
        static_count = _read_i32(data, base + static_info.rel_offset) or 0

    trans_tracks = trans_frames = 0
    trans_info = local_sections.get(3)
    if trans_info is not None and trans_info.size >= 8:
        trans_tracks, trans_frames, _trans_rate = _local_header(data, base + trans_info.rel_offset)

    anim_length = _read_f32(data, base + 0x74)
    q_a = _read_float4(data, base + 0x50)
    q_b = _read_float4(data, base + 0x60)

    notes: list[str] = []
    if _pop(mask_bits["animated_rotation"]) != rot_tracks:
        notes.append(
            "animated rotation mask popcount does not match local keyframe tracks "
            f"({_pop(mask_bits['animated_rotation'])}/{rot_tracks})"
        )
    if _pop(mask_bits["constant_rotation"]) != const_count:
        notes.append(
            "constant rotation mask popcount does not match local constant rotations "
            f"({_pop(mask_bits['constant_rotation'])}/{const_count})"
        )
    if static_count and _pop(mask_bits["static_translation"]) != static_count:
        notes.append(
            "static translation mask popcount does not match local static values "
            f"({_pop(mask_bits['static_translation'])}/{static_count})"
        )
    if trans_tracks and _pop(mask_bits["animated_translation"]) != trans_tracks:
        notes.append(
            "animated translation mask popcount does not match local vec3 tracks "
            f"({_pop(mask_bits['animated_translation'])}/{trans_tracks})"
        )

    return WeaponBlockMapping(
        name=name,
        slot=slot,
        rel_offset=rel_offset,
        abs_offset=base,
        size=payload_size,
        anim_length=anim_length,
        reference_rotation_a=q_a,
        reference_rotation_b=q_b,
        rotation_tracks=rot_tracks,
        frame_count=frame_count,
        sample_rate=sample_rate,
        constant_rotation_count=const_count,
        static_translation_count=static_count,
        animated_translation_tracks=trans_tracks,
        animated_translation_frames=trans_frames,
        animated=_selection(weapon_names, mask_bits["animated_rotation"]),
        constant=_selection(weapon_names, mask_bits["constant_rotation"]),
        static_translated=_selection(weapon_names, mask_bits["static_translation"]),
        animated_translated=_selection(weapon_names, mask_bits["animated_translation"]),
        sections=sections_by_name,
        notes=notes,
    )


def discover_weapon_blocks(data: bytes, sections: dict[str, LocalSectionInfo]) -> list[tuple[str, str, int, int, int]]:
    """Return [(name, slot, rel_from_unk5, abs_base, size), ...]."""
    unk5 = sections.get("UnkSec5")
    if unk5 is None or unk5.size < 0xA0:
        return []
    if not looks_like_weapon_container(data, unk5.abs_offset, unk5.size):
        return []

    out = [("weapon_block_0", "top-level UnkSec5", 0, unk5.abs_offset, unk5.size)]
    raw = local_section_table(data, unk5.abs_offset, unk5.size)
    if raw is None:
        return out

    local_sizes = local_section_sizes(raw, unk5.size)
    for idx in (6, 7):
        info = local_sizes.get(idx)
        if info is None or info.size < 0xA0:
            continue
        child_base = unk5.abs_offset + info.rel_offset
        if looks_like_weapon_container(data, child_base, info.size):
            out.append((
                f"weapon_block_{len(out)}",
                f"local slot {idx}",
                info.rel_offset,
                child_base,
                info.size,
            ))
    return out


def build_mapping(mab_path: Path, main_skeleton_path: Path,
                  weapon_skeleton_path: Path) -> MabMapping:
    data = mab_path.read_bytes()
    if not data or data[0] != FC2_VERSION:
        raise ValueError(f"not an FC2 MAB version 0x4C file: {mab_path}")

    raw_sections, sections = parse_sections(data)
    _raw_sections = raw_sections
    anim_length = _read_f32(data, FC2_ANIMLEN_OFF)
    if anim_length is None:
        raise ValueError("could not read FC2 animation length")
    rotation_tracks, frame_count, sample_rate = _section_header(data, sections, "Keyframes")

    main_names = read_lks_bone_names(main_skeleton_path)
    main_mapping = build_main_mapping(data, sections, main_skeleton_path, main_names)

    weapon_block_refs = discover_weapon_blocks(data, sections)
    weapon_blocks: list[WeaponBlockMapping] = []
    weapon_skeleton_used: str | None = None
    if weapon_block_refs:
        weapon_names = read_lks_bone_names(weapon_skeleton_path)
        weapon_skeleton_used = str(weapon_skeleton_path)
        for name, slot, rel, abs_base, size in weapon_block_refs:
            weapon_blocks.append(parse_weapon_block(
                data, weapon_names, abs_base, size, name, slot, rel
            ))

    return MabMapping(
        mab_path=str(mab_path),
        size=len(data),
        version="Far Cry 2 (0x4C)",
        anim_length=anim_length,
        frame_count=frame_count,
        sample_rate=sample_rate,
        sections=sections,
        main=main_mapping,
        weapon_skeleton_path=weapon_skeleton_used,
        weapon_blocks=weapon_blocks,
    )


def _print_selection(label: str, selection: BoneSelection, indent: str = "  ",
                     unit: str = "track") -> None:
    print(f"{indent}{label}: {selection.count}")
    if selection.names:
        for track, (bone_index, name) in enumerate(zip(selection.indices, selection.names)):
            print(f"{indent}  {unit} {track:02d} -> bone[{bone_index:02d}] {name}")


def print_report(mapping: MabMapping) -> None:
    print(f"MAB: {mapping.mab_path}")
    print(f"Size: {mapping.size} bytes")
    print(f"Version: {mapping.version}")
    print(f"Anim length: {mapping.anim_length:.6f}s")
    print(f"Frame count: {mapping.frame_count}")
    print(f"Sample rate/unknown: {mapping.sample_rate}")
    print("")

    print("Sections:")
    for label in SECTION_LABELS:
        info = mapping.sections.get(label)
        if info is None:
            print(f"  {label}: absent")
        else:
            print(
                f"  {label}: raw=0x{info.rel_offset:X} "
                f"abs=0x{info.abs_offset:X} size=0x{info.size:X}"
            )
    print("")

    main = mapping.main
    print("Main Skeleton Mapping")
    print(f"  Skeleton: {main.skeleton_path}")
    print(f"  Bone count: {main.bone_count}")
    print(
        "  Counts: "
        f"animated rotations={main.rotation_tracks}, "
        f"constant rotations={main.constant_rotations}, "
        f"translations={main.translation_tracks}"
    )
    print(f"  Translation source: {main.translation_source}")
    print(
        "  Translation source counts: "
        f"UnkSec3 static={main.static_translation_values}, "
        f"Offsets animated={main.animated_translation_tracks}"
    )
    print(f"  Rotation masks validate: {main.masks_validate}")
    print(f"  Translation mask validates: {main.translation_mask_validates}")
    _print_selection("Animated rotation bones", main.animated, unit="track")
    _print_selection("Constant rotation bones", main.constant, unit="const")
    _print_selection("Translated bones", main.translated, unit="track")
    for note in main.notes:
        print(f"  note: {note}")
    print("")

    if not mapping.weapon_blocks:
        print("Weapon Blocks: none detected in top-level UnkSec5")
        return

    print("Weapon Blocks")
    print(f"  Weapon skeleton: {mapping.weapon_skeleton_path}")
    for block in mapping.weapon_blocks:
        print("")
        print(f"  {block.name} ({block.slot})")
        print(
            f"    rel=0x{block.rel_offset:X} abs=0x{block.abs_offset:X} "
            f"size=0x{block.size:X}"
        )
        if block.anim_length is not None:
            print(f"    local anim length: {block.anim_length:.6f}s")
        print(
            "    counts: "
            f"animated rotations={block.rotation_tracks}, "
            f"constant rotations={block.constant_rotation_count}, "
            f"static translations={block.static_translation_count}, "
            f"animated translations={block.animated_translation_tracks}"
        )
        print(
            f"    local frames={block.frame_count}, "
            f"translation frames={block.animated_translation_frames}, "
            f"sample rate/unknown={block.sample_rate}"
        )
        print("    local sections:")
        for label, info in block.sections.items():
            print(
                f"      {label}: rel=0x{info.rel_offset:X} "
                f"abs=0x{info.abs_offset:X} size=0x{info.size:X}"
            )
        _print_selection("Animated rotation bones", block.animated, indent="    ", unit="track")
        _print_selection("Constant rotation bones", block.constant, indent="    ", unit="const")
        _print_selection("Static translation bones", block.static_translated, indent="    ", unit="value")
        _print_selection("Animated translation bones", block.animated_translated, indent="    ", unit="track")
        for note in block.notes:
            print(f"    note: {note}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Dump mapped bones and counts from one Far Cry 2 .mab file."
    )
    ap.add_argument("main_skeleton", type=Path, help="Character/main FC2 .skeleton file.")
    ap.add_argument("weapon_skeleton", type=Path, help="Weapon FC2 .skeleton file.")
    ap.add_argument("mab", type=Path, help="FC2 .mab file to inspect.")
    ap.add_argument("--json", type=Path, help="Write the same mapping to JSON.")
    args = ap.parse_args(argv)

    if not args.main_skeleton.exists():
        print(f"error: main skeleton does not exist: {args.main_skeleton}", file=sys.stderr)
        return 2
    if not args.mab.exists():
        print(f"error: MAB does not exist: {args.mab}", file=sys.stderr)
        return 2
    # Weapon skeleton is deliberately checked only after a weapon block exists.

    try:
        mapping = build_mapping(args.mab, args.main_skeleton, args.weapon_skeleton)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_report(mapping)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(asdict(mapping), indent=2), encoding="utf-8")
        print("")
        print(f"JSON written: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
