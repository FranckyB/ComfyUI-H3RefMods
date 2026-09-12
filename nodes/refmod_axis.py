"""
H3RefModApplyAxis — choose between two picker bundles with separate video/audio signed values
Think of it as creating a Slider from two RefMods
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from ..py.refmod_common import (
    _prompt_hint,
)

from ..py.refmod_core import (
    H3RefMod,
)
_PACK_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_MODS_DIR = os.path.join(_PACK_DIR, "mods")  # pre-models/refmods storage, still read


# reference retention presets (master strength multiplier on Apply)
RETENTION = {
    "fully_preserved": 1.0,
    "partially_preserved": 0.7,
    "attribute_transfer": 0.4,
    "weak_reference": 0.15,
}

def _info_lines(mod):
    opt = mod.optimize_steps
    if mod.mode == "encode":
        opt = f"n/a ({mod.optimize_steps} — encode mode stores the actual encode)"
    return [
        "=" * 52,
        f"  MiniMax H3 RefMod: {mod.name}",
        f"  {'concept_type':<18} {mod.concept_type}",
        f"  {'mode':<18} {mod.mode}",
        f"  {'kind':<18} {mod.kind}",
        f"  {'latent':<18} {tuple(mod.latent.shape)}",
        f"  {'tokens injected':<18} {mod.token_count}",
        f"  {'source':<18} {mod.source} ({mod.source_shape})",
        f"  {'pool':<18} {mod.pool}",
        f"  {'identity':<18} {opt}",
        f"  {'tags':<18} {', '.join(mod.tags) if mod.tags else '-'}",
        f"  {'description':<18} {mod.description or '-'}",
        "=" * 52,
    ]

def _mod_group_key(mod: H3RefMod) -> str:
    name = str(getattr(mod, "name", "") or "")
    lower_name = name.lower()
    for suffix in ("_Video", "_Audio"):
        if lower_name.endswith(suffix.lower()):
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
    CATEGORY = "H3RefModPicker"

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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModApplyAxis": "Apply H3 RefMod Axis",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
