"""
H3RefModApplySimple — identity-oriented RefMod apply node with one strength dial
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Dict, List

from comfy_api.latest import io

# reference retention presets (master strength multiplier on Apply)
RETENTION = {
    "fully_preserved": 1.0,
    "partially_preserved": 0.7,
    "attribute_transfer": 0.4,
    "weak_reference": 0.15,
}


def _ref_blocks(mods, retention, curve=None, seed=-1,
                use_video: bool = True, use_audio: bool = True) -> List[Dict]:
    """Ref blocks for a loader bundle, scaled by row strength x retention.

    ``retention`` is a master strength multiplier: a float 0-1 (1.0 =
    fully_preserved, 0.7 = partially_preserved, 0.4 = attribute_transfer,
    0.15 = weak_reference), or one of those preset names for legacy
    workflows saved with the old combo widget.

    ``curve`` (optional) is a per-frame strength spec — a ``(direction,
    shape, value)`` tuple, a legacy preset name, per-frame values, control
    points (see ``core.curve_strengths``) — applied on top of the row
    strength.  A flat/no curve keeps today's behavior.

    ``seed`` (default -1 = off) enables ref scrambling: with 2+ refs in the
    bundle, the order is shuffled and a random subset kept, so a different
    ref leads each run instead of the same one always "popping".  Same seed
    -> same scramble; connect/randomize the seed for per-run variation.
    """
    if isinstance(retention, str):
        factor = RETENTION.get(retention, 1.0)
    else:
        factor = float(retention)
    items = [] if mods is None else list(mods)
    if int(seed) >= 0 and len(items) > 1:
        rng = random.Random(int(seed))
        rng.shuffle(items)
        keep = rng.randint(max(1, len(items) // 2), len(items))
        items = items[:keep]
        print(f"[H3RefModApply] scramble seed={int(seed)}: "
              f"{len(mods)} refs -> kept {len(items)} (order shuffled)")
    blocks = []
    debug_rows = []
    for mod, strength in items:
        eff = min(1.0, max(0.0, strength * factor))
        block = mod.ref_block(eff, curve=curve, use_video=use_video, use_audio=use_audio)
        if block is not None:
            blocks.append(block)
            debug_rows.append(f"{mod.name}@{eff:.2f}")
        else:
            debug_rows.append(f"{mod.name}@{eff:.2f} (skipped)")
    if debug_rows:
        print("[H3RefModApply] effective refs: " + ", ".join(debug_rows))
    return blocks


class H3RefModApplySimple(io.ComfyNode):
    """Identity-oriented RefMod apply node with one strength dial."""

    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template(
            "cond",
            allowed_types=[io.Custom("MINIMAX_H3_COND"), io.Conditioning])
        return io.Schema(
            node_id="H3RefModApplySimple",
            display_name="Apply H3 RefMod",
            description=(
                "Apply one or more RefMods to a MiniMax H3 conditioning with a single "
                "Strength control. This identity-oriented version uses a flat full-length "
                "reference curve under the hood."
            ),
            category="H3RefMod",
            inputs=[
                io.MatchType.Input("conditioning", template=template,
                    tooltip="MINIMAX_H3_COND (ComfyUI-MiniMaxH3 pack) or CONDITIONING "
                            "(core MiniMaxH3ReferenceToVideo)."),
                io.Custom("H3_REF_MODS").Input("mods",
                    optional=True,
                    tooltip="Bundle of RefMods to inject. Leave unconnected to bypass unchanged."),
                io.Boolean.Input("use_video", default=True,
                    tooltip="Apply the visual latent from each RefMod. Turn this off to use only embedded audio."),
                io.Boolean.Input("use_audio", default=True,
                    tooltip="Apply the embedded audio latent from each RefMod when present. Turn this off for visual-only application."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Master reference strength. For identity work this is the main "
                            "dial: higher = tighter identity lock, lower = more freedom but "
                            "more drift."),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="conditioning",
                    tooltip="The conditioning with the ref blocks injected, same type as the input."),
            ],
        )

    @classmethod
    def execute(cls, conditioning, mods, use_video=True, use_audio=True, strength=1.0):
        if mods is None:
            print("[H3RefModApplySimple] no mods connected — bypassing conditioning unchanged")
            return io.NodeOutput(conditioning)
        if not use_video and not use_audio:
            print("[H3RefModApplySimple] use_video=False and use_audio=False — bypassing conditioning unchanged")
            return io.NodeOutput(conditioning)
        blocks = _ref_blocks(mods, strength, ("constant", "linear", 1.0), seed=-1,
                             use_video=bool(use_video), use_audio=bool(use_audio))
        if isinstance(conditioning, list):
            out = []
            for t in conditioning:
                d = dict(t[1])
                d["minimax_refs"] = list(d.get("minimax_refs", [])) + blocks
                out.append([t[0], d])
            print(f"[H3RefModApplySimple] strength={strength} "
                  f"({len(blocks)} ref block(s) injected)")
        else:
            out = replace(conditioning, refs=list(conditioning.refs) + blocks)
            print(f"[H3RefModApplySimple] strength={strength} "
                  f"({len(blocks)} ref block(s) injected, {len(out.refs)} total)")
        return io.NodeOutput(out)


NODE_CLASS_MAPPINGS = {
    "H3RefModApplySimple":    H3RefModApplySimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModApplySimple":    "Apply H3 RefMod Simple",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
