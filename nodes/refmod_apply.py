"""
nodes.py — ComfyUI nodes for MiniMax H3 "RefMod" (no-training reference mods)

  H3RefModFolderLoader  — load every image/video in a folder as an ordered ref list
  H3RefModApply         — inject the bundle into a MINIMAX_H3_COND conditioning or the
                          built-in ComfyUI CONDITIONING (one node, old ApplyCond
                          workflows auto-migrate via node replacement)

Mods are stored in ``models/refmods/`` (created on first run, next to loras/
and unet/); mods saved by older versions in the pack's ``mods/`` folder still
load.

The mod rides the model's native ref2va path: the Apply nodes append reference
blocks (the mod latents) to the conditioning's ``refs``, and the DiT attends
to those tokens through all of its blocks, exactly like a full image/video
reference but at a fraction of the token budget.

Reference strength uses the model's own conditioning-strength dial, but
weakening a ref mixes its latent toward a heavily blurred copy of itself
(not toward noise or toward zero — see core.py's ``_blur_latent``/
``ref_block`` for why). ``retention`` on the Apply nodes is a preset master
strength (fully_preserved / partially_preserved / attribute_transfer /
weak_reference) multiplied with each loader row's strength.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

import comfy.patcher_extension
import comfy.utils
import folder_paths
from comfy_api.latest import io

from ..py.refmod_common import (
    list_media_files,
    load_image_file,
    load_video_file,
    refmods_dir,
)
from ..py.refmod_core import (
    CURVE_DIRECTIONS,
    CURVE_SHAPES,
    _blur_latent,
    curve_value_at,
)
from ..py.debug_grid import (
    graph_pnginfo,
    pil_to_tensor,
    read_graph_meta,
    render_debug_grid,
)

# reference retention presets (master strength multiplier on Apply)
RETENTION = {
    "fully_preserved": 1.0,
    "partially_preserved": 0.7,
    "attribute_transfer": 0.4,
    "weak_reference": 0.15,
}


# ═══════════════════════════════════════════════════════════════════════════
# Graph presets (shared curve files, next to the mods)
# ═══════════════════════════════════════════════════════════════════════════

def _graph_presets_dir() -> str:
    """models/refmods/graph_presets — shared curve presets, created on first use."""
    d = os.path.join(refmods_dir(), "graph_presets")
    os.makedirs(d, exist_ok=True)
    return d


def _list_graph_presets() -> List[str]:
    """Graph preset names in the presets folder, for the dropdown.

    Presets are PNGs with the graph embedded in their tEXt metadata (a saved
    debug grid); legacy .json files from before the switch still list.  PNGs
    without a valid graph chunk are skipped so random images dropped in the
    folder don't show up.
    """
    d = _graph_presets_dir()
    try:
        entries = sorted(os.listdir(d))
    except OSError:
        return []
    names = []
    for fn in entries:
        if fn.endswith(".json"):
            names.append(fn[:-5])
        elif fn.endswith(".png") and read_graph_meta(os.path.join(d, fn)) is not None:
            names.append(fn[:-4])
    return names


def _load_graph_preset(name: str) -> Optional[tuple]:
    """Read a graph preset -> (direction, shape, value) or None if invalid.

    Presets are PNG files with the graph embedded in their tEXt metadata (the
    saved debug grid — share the image itself); legacy .json presets still
    load.
    """
    d = _graph_presets_dir()
    meta = read_graph_meta(os.path.join(d, name + ".png"))
    if meta is not None:
        return meta
    try:
        with open(os.path.join(d, name + ".json"), "r",
                  encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    direction, shape = data.get("direction"), data.get("shape")
    if direction not in CURVE_DIRECTIONS or shape not in CURVE_SHAPES:
        return None
    try:
        value = float(data.get("value", 1.0))
    except (TypeError, ValueError):
        return None
    return (direction, shape, value)


def _save_graph_preset(name: str, spec, img=None) -> str:
    """Write a (direction, shape, value) tuple as a PNG preset with tEXt meta.

    The saved file is the debug grid itself (a mini preview of the curve)
    with the graph embedded in its metadata, so sharing the image shares the
    curve.  ``img`` is the rendered grid from Apply; when absent a minimal
    grid is rendered just for the file.
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "_"
                    for c in str(name).strip())
    if not safe:
        return ""
    if img is None:
        img = render_debug_grid(spec)
    img.save(os.path.join(_graph_presets_dir(), safe + ".png"),
             pnginfo=graph_pnginfo(spec))
    return safe


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


def _make_step_wrapper(spec) -> Callable:
    """DIFFUSION_MODEL wrapper re-mixing every ref latent per denoising step.

    The Apply node's frame curve is baked into the ref latents once, before
    sampling.  This wrapper re-scales them per step instead: it caches the
    pristine latents (and blurred copies) on the first forward, then each
    step mixes toward the blur with ``curve_value_at(spec, 1 - sigma)`` — the
    same direction/shape/value envelope as the frame curve, running over the
    denoise timeline (0 = first step, high sigma) instead of the video's.
    ``cond_video_latents`` is re-read from the payload every forward, so
    replacing the list per step is all it takes; the packed layout stays
    valid because only values change, never shapes.
    """
    state = {"pristine": None, "blurred": None}

    def wrapper(executor, x, timestep, context, transformer_options, **kwargs):
        payload = kwargs.get("minimax_payload") or {}
        cond = payload.get("cond_video_latents")
        if cond:
            if state["pristine"] is None:
                state["pristine"] = [z.clone() for z in cond]
                state["blurred"] = [_blur_latent(z) for z in state["pristine"]]
            sigma = float((timestep.flatten()[0] / 1000.0).clamp(0.0, 1.0))
            s = curve_value_at(spec, 1.0 - sigma)
            if s >= 1.0:
                payload["cond_video_latents"] = state["pristine"]
            else:
                s = max(0.0, min(1.0, s))
                payload["cond_video_latents"] = [
                    s * p + (1.0 - s) * b
                    for p, b in zip(state["pristine"], state["blurred"])
                ]
        return executor(x, timestep, context, transformer_options, **kwargs)

    return wrapper

# ═══════════════════════════════════════════════════════════════════════════
# Node: H3RefModApplyAdvanced / ApplyCond
# ═══════════════════════════════════════════════════════════════════════════

class H3RefModApplyAdvanced(io.ComfyNode):
    """
    Inject a loader bundle of RefMods into a MiniMax H3 conditioning.

    Accepts either the ComfyUI-MiniMaxH3 pack's MINIMAX_H3_COND or the built-in
    ComfyUI CONDITIONING (from the core MiniMaxH3ReferenceToVideo node, or
    MiniMax H3 RefMods to Video) and returns the same type.  Appends each
    mod's reference latent to the conditioning's ``refs`` / ``minimax_refs``,
    so the DiT attends to it through all blocks exactly like a reference
    image/video — the same content-based binding MiniMax H3 RefMods to Video
    uses internally.  Use this node instead when you need to chain onto an
    existing conditioning (e.g. the native ref2va node) or want the per-run
    strength curve / scramble features below, which RefMods to Video doesn't
    have.  ``retention`` is
    a master strength over the loader's per-row strengths; the curve is split
    into ``curve_direction`` (constant / concept_at_start / concept_at_end),
    ``curve_shape`` (how the envelope travels between its endpoints) and
    ``curve_value`` (the non-zero endpoint) — all plain widgets, no ComfyUI
    Curve widget required.  ``scramble_seed`` (default -1 = off) shuffles the
    ref order and keeps a random subset per run so a multi-ref mod can "pop"
    a different ref each time instead of always the same one.
    """

    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template(
            "cond",
            allowed_types=[io.Custom("MINIMAX_H3_COND"), io.Conditioning])
        return io.Schema(
            node_id="H3RefModApplyAdvanced",
            display_name="Apply H3 RefMod Advanced",
            description=(
                "Inject a loader bundle of RefMods into a MiniMax H3 conditioning. "
                "Accepts both the pack's MINIMAX_H3_COND and the built-in "
                "CONDITIONING and returns the same type."
            ),
            category="H3RefMod",
            inputs=[
                io.MatchType.Input("conditioning", template=template,
                    tooltip="MINIMAX_H3_COND (ComfyUI-MiniMaxH3 pack) or CONDITIONING "
                            "(core MiniMaxH3ReferenceToVideo)."),
                io.Custom("H3_REF_MODS").Input("mods",
                    optional=True,
                    tooltip="Bundle from Load RefMod / Load RefMod Stack / Visual RefMod Picker / "
                            "Create H3 RefMod. Leave unconnected to bypass unchanged."),
                io.Boolean.Input("use_video", default=True,
                    tooltip="Apply the visual latent from each RefMod. Turn this off to use only embedded audio."),
                io.Boolean.Input("use_audio", default=True,
                    tooltip="Apply the embedded audio latent from each RefMod when present. Turn this off for visual-only application."),
                io.Float.Input("retention", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Master reference strength, multiplied with each loader row's "
                             "strength. MiniMax retention levels: 1.0 = fully_preserved, "
                             "0.7 = partially_preserved, 0.4 = attribute_transfer (keep "
                             "style/attributes, not identity), 0.15 = weak_reference. "
                             "0 = no reference."),
                io.Combo.Input("curve_direction", options=list(CURVE_DIRECTIONS),
                    default="concept_at_end",
                    tooltip="Where the concept shows up in the output (the mirror of the ref's "
                            "strength envelope): 'concept_at_end' (default, was 'decrease') locks "
                            "the ref's literal footage in at the START and releases it toward the "
                            "end — the identity/character emerges in the second half, without "
                            "dragging the ref's background in; 'concept_at_start' (was 'increase') "
                            "opens free from the ref and locks onto it near the END — the concept "
                            "shows early; 'concept_at_middle' peaks mid-video ([0..1..0] — the "
                            "concept appears only in the middle); 'concept_at_ends' holds both "
                            "ends with a mid dip ([1..0..1]); 'constant' keeps one strength for "
                            "the whole video (flat at curve_value = 1.0, today's behavior). Old "
                            "'decrease'/'increase' values saved in workflows still resolve."),
                io.Int.Input("scramble_seed", default=-1, min=-1, max=2147483647, step=1,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip="Ref scrambling seed. -1 (default) = off: all refs in saved order. "
                            "With 2+ refs in the bundle, a seed >= 0 shuffles the ref order and "
                            "keeps a random subset, so a different ref leads each run (a multi-ref "
                            "mod 'pops' a different video/image per seed). Same seed = same "
                            "scramble; set this widget's control-after-generate to 'randomize' "
                            "for per-run variation."),
                io.Combo.Input("curve_shape", options=list(CURVE_SHAPES),
                    default="ease",
                    tooltip="How the envelope travels between its endpoints: 'ease' (smoothstep, "
                            "default), 'linear', 'sigmoid'/'tanh' (smooth S-curves, tanh with a "
                            "steeper knee), 'quadratic', 'cubic', 'exponential', 'stair' "
                            "(stepped), 'elastic' (overshoots), 'bump' (peak mid-video, for "
                            "one specific action), 'dip' (trough mid-video)."),
                io.Float.Input("curve_value", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Endpoint value ('user input'): both endpoints for 'constant' and "
                            "'concept_at_ends', the end for 'concept_at_start', the start for "
                            "'concept_at_end', the mid peak for 'concept_at_middle'. "
                            "1.0 = full strength there."),
                io.Combo.Input("graph_preset",
                    options=["(none)"] + _list_graph_presets(), default="(none)",
                    optional=True,
                    tooltip="Optional shared graph preset — leave on '(none)' to use the curve "
                            "widgets above. Selecting one loads direction/shape/value from a "
                            "saved debug-grid PNG (graph embedded in its metadata) or a legacy "
                            ".json, in models/refmods/graph_presets/. Share the preset PNG "
                            "itself to share a curve. New presets appear after a restart."),
                io.String.Input("save_preset_as", default="", optional=True,
                    tooltip="Optional: type a name and run to save the current (resolved) curve "
                            "as a PNG preset — the curve graph itself with the graph embedded in "
                            "its metadata — in models/refmods/graph_presets/. Share that image "
                            "to share the curve. Leave empty to skip."),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="conditioning",
                    tooltip="The conditioning with the ref blocks injected, same type as the input."),
                io.Image.Output("debug", display_name="curve graph",
                    tooltip="Optional 1024x1024 curve graph: the strength envelope "
                            "(direction/shape/value) with the concept zone shaded. Leave "
                            "unconnected to skip the preview."),
            ],
        )

    @classmethod
    def execute(cls, conditioning, mods, use_video=True, use_audio=True, retention=1.0,
                curve_direction="concept_at_end", curve_shape="ease", curve_value=1.0,
                strength_curve=None, scramble_seed=-1, graph_preset="", save_preset_as=""):
        if mods is None:
            img = render_debug_grid((curve_direction, curve_shape, curve_value), "")
            print("[H3RefModApply] no mods connected — bypassing conditioning unchanged")
            return io.NodeOutput(conditioning, pil_to_tensor(img))
        if not use_video and not use_audio:
            img = render_debug_grid((curve_direction, curve_shape, curve_value), "")
            print("[H3RefModApply] use_video=False and use_audio=False — bypassing conditioning unchanged")
            return io.NodeOutput(conditioning, pil_to_tensor(img))
        # workflows saved before the curve split pass the old single preset name
        curve = strength_curve if strength_curve is not None \
            else (curve_direction, curve_shape, curve_value)
        # a selected graph preset overrides the curve widgets
        preset_name = ""
        if graph_preset and graph_preset != "(none)":
            loaded = _load_graph_preset(graph_preset)
            if loaded is None:
                print(f"[H3RefModApply] WARNING: graph preset '{graph_preset}' "
                      f"not found or invalid — using widget curve")
            else:
                curve = loaded
                preset_name = graph_preset
        img = render_debug_grid(curve, preset_name)
        if save_preset_as:
            saved = _save_graph_preset(save_preset_as, curve, img)
            if saved:
                print(f"[H3RefModApply] graph preset saved: {saved}.png "
                      f"({curve[0]} + {curve[1]} @ {float(curve[2]):.2f})")
        blocks = _ref_blocks(mods, retention, curve, seed=scramble_seed,
                     use_video=bool(use_video), use_audio=bool(use_audio))
        if isinstance(conditioning, list):
            # built-in ComfyUI CONDITIONING (core MiniMaxH3ReferenceToVideo)
            out = []
            for t in conditioning:
                d = dict(t[1])
                d["minimax_refs"] = list(d.get("minimax_refs", [])) + blocks
                out.append([t[0], d])
            print(f"[H3RefModApply] retention={retention} "
                  f"({len(blocks)} ref block(s) injected)")
        else:
            # ComfyUI-MiniMaxH3 pack MINIMAX_H3_COND
            out = replace(conditioning, refs=list(conditioning.refs) + blocks)
            print(f"[H3RefModApply] retention={retention} "
                  f"({len(blocks)} ref block(s) injected, {len(out.refs)} total)")
        return io.NodeOutput(out, pil_to_tensor(img))


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


# ═══════════════════════════════════════════════════════════════════════════
# Node: H3RefModStepCurve
# ═══════════════════════════════════════════════════════════════════════════

class H3RefModApplyStepCurve:
    """Per-step (per-sigma) reference strength curve, applied at generation time.

    The Apply node's frame curve is baked into the ref latent once, before
    sampling.  This node instead re-mixes every ref latent once per denoising
    step: early steps (high sigma) set global structure and identity, late
    steps (low sigma) paint fine texture — so the same direction/shape/value
    envelope runs over the denoise timeline instead of the video's.  Same
    curve widgets as Apply; attach between the model loader and the sampler
    (MODEL -> MODEL, same type).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "The H3 model to patch. Returned unchanged apart from the per-step "
                               "ref-mixing wrapper."}),
                "curve_direction": (list(CURVE_DIRECTIONS), {"default": "concept_at_end",
                    "tooltip": "Same envelope as the Apply frame curve, but over the DENOISE "
                               "timeline: 'concept_at_end' (default) keeps the refs at full "
                               "strength in the early steps (high sigma — composition and identity "
                               "set first) and releases them toward the final steps (clean texture, "
                               "no ref grain); 'concept_at_start' opens weak and locks full strength "
                               "in the late steps (identity detail refined at the end); 'constant' "
                               "keeps one strength for every step; 'concept_at_middle' peaks "
                               "mid-denoise; 'concept_at_ends' holds the extremes and dips "
                               "mid-denoise. Old 'decrease'/'increase' values still resolve."}),
                "curve_shape": (list(CURVE_SHAPES), {"default": "ease",
                    "tooltip": "How the per-step strength travels between its endpoints (same "
                               "shapes as the Apply frame curve: linear / ease / sigmoid / tanh / "
                               "quadratic / cubic / exponential / stair / elastic / bump / dip)."}),
                "curve_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "number",
                    "tooltip": "Endpoint strength ('user input'): both endpoints for 'constant' and "
                               "'concept_at_ends', the end for 'concept_at_start', the start for "
                               "'concept_at_end', the mid peak for 'concept_at_middle'. "
                               "1.0 = full ref there."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "H3RefMod"

    def apply(self, model, curve_direction="concept_at_end",
              curve_shape="ease", curve_value=1.0):
        model = model.clone()
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "h3_apply_refmod_step_curve",
            _make_step_wrapper((curve_direction, curve_shape, curve_value)))
        print(f"[H3RefModApplyStepCurve] {curve_direction} + {curve_shape} "
              f"@ {curve_value:.2f} attached — refs re-mixed per denoising step")
        return (model,)




# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "H3RefModApplySimple":    H3RefModApplySimple,
    "H3RefModApplyAdvanced":  H3RefModApplyAdvanced,
    "H3RefModApplyStepCurve": H3RefModApplyStepCurve,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModApplySimple":    "Apply H3 RefMod",
    "H3RefModApplyAdvanced":  "Apply H3 RefMod Advanced",
    "H3RefModApplyStepCurve": "Apply H3 RefMod Step Curve",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
