"""
nodes.py — ComfyUI nodes for MiniMax H3 "RefMod" (no-training reference mods)

  H3RefModsAxis         — A/B mod pairs on one signed slider each (negative -> A, positive -> B)

"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple
import folder_paths


from ..py.refmod_common import (
    refmods_dir,
    _prompt_hint,
    _info_lines
)

from ..py.refmod_core import (
    H3RefMod,
    read_refmod_meta,
)

from ..py.debug_grid import (
    read_graph_meta,
)

_PACK_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_MODS_DIR = os.path.join(_PACK_DIR, "mods")  # pre-models/refmods storage, still read
_MOD_CACHE: Dict[str, H3RefMod] = {}
_MOD_CACHE_MAX = 24          # cap: never pin more mods in RAM than this (FIFO eviction)
_MOD_LIST_CACHE_KEY = None   # (dirs, mtimes, sizes) signature of the last _list_mod_names() scan
_MOD_LIST_CACHE_VAL = None
_MOD_SKIP_DIRS = {"graph_presets", ".git", "__pycache__"}


# reference retention presets (master strength multiplier on Apply)
RETENTION = {
    "fully_preserved": 1.0,
    "partially_preserved": 0.7,
    "attribute_transfer": 0.4,
    "weak_reference": 0.15,
}

# ═══════════════════════════════════════════════════════════════════════════
# Mods folder helpers
# ═══════════════════════════════════════════════════════════════════════════


def _iter_mod_paths(base_dir: str):
    """Yield ``(relative_stem, absolute_stem)`` for RefMods under ``base_dir``."""
    for root, dirnames, filenames in os.walk(base_dir):
        dirnames[:] = sorted(d for d in dirnames if d not in _MOD_SKIP_DIRS)
        for fn in sorted(filenames):
            if not fn.endswith(".safetensors"):
                continue
            abs_stem = os.path.join(root, fn[:-len(".safetensors")])
            rel_stem = os.path.relpath(abs_stem, base_dir).replace(os.sep, "/")
            yield rel_stem, abs_stem


def _list_mod_names() -> List[str]:
    """Available RefMod names across the search dirs (for the loader dropdown).

    Only entries with valid RefMod metadata (embedded in the safetensors header
    or a legacy sidecar .json) are listed, so other mod formats in
    models/mods/ (e.g. LTXMod files) don't show up.

    Called by INPUT_TYPES/VALIDATE_INPUTS on every prompt validation, so the
    result is cached until any mod file appears/disappears/changes (checked
    via cheap os.stat, not by re-reading every safetensors header).
    """
    global _MOD_LIST_CACHE_KEY, _MOD_LIST_CACHE_VAL
    mods_dir = refmods_dir()
    sig = []
    if os.path.isdir(mods_dir):
        for rel_stem, abs_stem in _iter_mod_paths(mods_dir):
            try:
                st = os.stat(abs_stem + ".safetensors")
                sig.append(f"{mods_dir}:{rel_stem}:{st.st_size}:{int(st.st_mtime)}")
            except OSError:
                pass
    key = "\n".join(sig)
    if key == _MOD_LIST_CACHE_KEY:
        return _MOD_LIST_CACHE_VAL
    names = set()
    if os.path.isdir(mods_dir):
        for rel_stem, abs_stem in _iter_mod_paths(mods_dir):
            meta = read_refmod_meta(abs_stem)
            if meta is not None and meta.get("kind") in ("image", "video", "audio"):
                names.add(rel_stem)
    _MOD_LIST_CACHE_KEY, _MOD_LIST_CACHE_VAL = key, sorted(names)
    return _MOD_LIST_CACHE_VAL


def _find_mod_path(name: str) -> str:
    mods_dir = refmods_dir()
    p = os.path.join(mods_dir, name)
    if os.path.isfile(p + ".safetensors"):
        return p
    raise FileNotFoundError(
        f"RefMod '{name}' not found. Searched:\n" +
        f"  - {mods_dir}/{name}.safetensors")


def _load_mod(name: str) -> H3RefMod:
    if name in _MOD_CACHE:
        return _MOD_CACHE[name]
    mod = H3RefMod.load(_find_mod_path(name), device="cpu")
    _MOD_CACHE[name] = mod
    if len(_MOD_CACHE) > _MOD_CACHE_MAX:
        # FIFO eviction: pop the oldest-loaded mod so a long session loading
        # many different mods doesn't accumulate every one of them in RAM
        _MOD_CACHE.pop(next(iter(_MOD_CACHE)))
    return mod


def _mod_group_key(mod: H3RefMod) -> str:
    name = str(getattr(mod, "name", "") or "")
    for suffix in ("_Video", "_Audio"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def _split_trailing_selection(rows, label: str) -> Tuple[List[Tuple[H3RefMod, float]], Optional[Tuple[H3RefMod, float]], Optional[Tuple[H3RefMod, float]]]:
    """Return the last logical RefMod selection from a bundle."""
    items = [] if rows is None else list(rows)
    if not items:
        return [], None, None
    last_key = _mod_group_key(items[-1][0])
    start = len(items) - 1
    while start > 0 and _mod_group_key(items[start - 1][0]) == last_key:
        start -= 1
    group = items[start:]
    if start > 0:
        print(f"[H3RefModApplyAxis] warning: {label} input contains {start} earlier row(s); using only the last RefMod selection '{last_key}'.")
    video_row = None
    audio_row = None
    for row in group:
        mod, _strength = row
        if mod.kind == "audio":
            audio_row = row
        else:
            video_row = row
    return group, video_row, audio_row


# ═══════════════════════════════════════════════════════════════════════════
# Node: H3RefModLoadAxis (signed A/B sliders)
# ═══════════════════════════════════════════════════════════════════════════

class H3RefModLoadAxis:
    """A/B mod pairs on one signed slider each.

    Each row has an A-side mod, a B-side mod and one ``value`` slider in
    [-1, 1]: negative values use the A mod, positive values use the B mod, and
    the magnitude is the reference strength (same 0-1 math as the loader).  A
    value of 0 skips the row entirely.  This makes concept axes like "young
    <-> old" or "clean <-> weathered" a single dial: extract the two extremes
    once, then slide between them.
    """

    MAX_SLOTS = 8
    NONE = "(none)"

    @classmethod
    def INPUT_TYPES(cls):
        names = [cls.NONE] + _list_mod_names()
        required = {
            "show_info": ("BOOLEAN", {"default": False,
                "tooltip": "Print the selected A/B pairs and strengths to the console."}),
        }
        for i in range(1, cls.MAX_SLOTS + 1):
            required[f"mod_a_{i}"] = (names, {"tooltip": f"A-side RefMod {i} (used when value_{i} is negative), or {cls.NONE}."})
            required[f"mod_b_{i}"] = (names, {"tooltip": f"B-side RefMod {i} (used when value_{i} is positive), or {cls.NONE}."})
            required[f"value_{i}"] = ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0,
                "step": 0.01, "display": "number",
                "tooltip": "Signed strength: negative uses mod_a, positive uses mod_b, 0 skips the "
                           "row. The magnitude is the reference strength (same 0-1 math as "
                           "Load H3 RefMods), so -0.5 injects mod_a at half strength."})
        return {"required": required}

    RETURN_TYPES = ("H3_REF_MODS", "STRING")
    RETURN_NAMES = ("mods", "prompt_hint")
    FUNCTION = "load"
    CATEGORY = "H3RefMod"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        available = set(_list_mod_names())
        for i in range(1, cls.MAX_SLOTS + 1):
            for side in ("a", "b"):
                name = str(kwargs.get(f"mod_{side}_{i}", cls.NONE))
                if name and name != cls.NONE and name not in available:
                    return (f"RefMod slot {i} ({side}): '{name}' not found in mods/. "
                            "Run Extract H3 RefMod first.")
        return True

    def load(self, show_info=False, **kwargs):
        loads = []
        for i in range(1, self.MAX_SLOTS + 1):
            value = float(kwargs.get(f"value_{i}", 0.0))
            if abs(value) < 1e-6:
                continue
            side = "b" if value > 0 else "a"
            name = str(kwargs.get(f"mod_{side}_{i}", self.NONE))
            if not name or name == self.NONE:
                continue
            loads.append((_load_mod(name), min(1.0, abs(value))))
        if loads:
            print("[H3RefModsAxis] " + ", ".join(
                f"{m.name}@{s:+.2f}" for m, s in loads)
                + f" ({sum(m.token_count for m, _ in loads)} tokens total)")
        else:
            print("[H3RefModsAxis] no rows selected (values 0 or both sides (none))")
        if show_info:
            for mod, strength in loads:
                print("\n".join(_info_lines(mod)))
                print(f"  {'strength':<18} {strength:.2f}")
        hint = _prompt_hint(loads)
        if hint:
            print(f"[H3RefModsAxis] prompt_hint: {hint}")
        return (loads, hint)


class H3RefModApplyAxis:
    """Choose between two picker bundles with separate video/audio signed values."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mods_a": ("H3_REF_MODS", {
                    "tooltip": "RefMod bundle from one Visual RefMod Picker. If older chained selections are present too, only the last logical RefMod selection is used.",
                }),
                "mods_b": ("H3_REF_MODS", {
                    "tooltip": "RefMod bundle from another Visual RefMod Picker. If older chained selections are present too, only the last logical RefMod selection is used.",
                }),
                "video_value": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "display": "number",
                    "tooltip": "Signed video selector: negative uses A video, positive uses B video. Magnitude becomes the output video strength. 0 skips video.",
                }),
                "audio_value": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "display": "number",
                    "tooltip": "Signed audio selector: negative uses A audio, positive uses B audio. If only one side has audio, that audio is used whenever this is non-zero.",
                }),
                "show_info": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print the selected output RefMods and strengths to the console.",
                }),
            },
        }

    RETURN_TYPES = ("H3_REF_MODS", "STRING")
    RETURN_NAMES = ("mods", "prompt_hint")
    FUNCTION = "select"
    CATEGORY = "H3RefMod"

    def select(self, mods_a, mods_b, video_value=0.0, audio_value=0.0, show_info=False):
        _group_a, video_a, audio_a = _split_trailing_selection(mods_a, "A")
        _group_b, video_b, audio_b = _split_trailing_selection(mods_b, "B")

        loads: List[Tuple[H3RefMod, float]] = []

        def pick_signed(value: float, row_a, row_b, kind: str):
            mag = min(1.0, abs(float(value)))
            if mag < 1e-6:
                return None
            chosen = row_b if value > 0 else row_a
            chosen_side = "B" if value > 0 else "A"
            fallback = row_a if value > 0 else row_b
            fallback_side = "A" if value > 0 else "B"
            if chosen is None:
                if fallback is not None:
                    print(f"[H3RefModApplyAxis] warning: {kind} {chosen_side} is missing; using {fallback_side} {kind} at {mag:.2f}.")
                    return fallback[0], mag
                print(f"[H3RefModApplyAxis] warning: no {kind} RefMod available on either side.")
                return None
            return chosen[0], mag

        video_pick = pick_signed(video_value, video_a, video_b, "video")
        if video_pick is not None:
            loads.append(video_pick)

        if audio_a is not None and audio_b is not None:
            audio_pick = pick_signed(audio_value, audio_a, audio_b, "audio")
        elif audio_a is not None or audio_b is not None:
            mag = min(1.0, abs(float(audio_value)))
            if mag < 1e-6:
                audio_pick = None
            else:
                only = audio_a or audio_b
                side = "A" if audio_a is not None else "B"
                print(f"[H3RefModApplyAxis] audio exists only on {side}; using that audio at {mag:.2f}.")
                audio_pick = (only[0], mag)
        else:
            audio_pick = None
        if audio_pick is not None:
            loads.append(audio_pick)

        if loads:
            print("[H3RefModApplyAxis] " + ", ".join(
                f"{mod.name}@{strength:.2f}" for mod, strength in loads
            ))
        else:
            print("[H3RefModApplyAxis] no output refs selected")
        if show_info:
            for mod, strength in loads:
                print("\n".join(_info_lines(mod)))
                print(f"  {'strength':<18} {strength:.2f}")
        hint = _prompt_hint(loads)
        if hint:
            print(f"[H3RefModApplyAxis] prompt_hint: {hint}")
        return (loads, hint)

# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "H3RefModApplyAxis": H3RefModApplyAxis,
    "H3RefModLoadAxis":  H3RefModLoadAxis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModApplyAxis": "Apply H3 RefMod Axis",
    "H3RefModLoadAxis":  "Load H3 RefMod Axis",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
