"""
nodes.py — ComfyUI nodes for MiniMax H3 "RefMod" (no-training reference mods)

  H3RefModFolderLoader  — load every image/video in a folder as an ordered ref list
  H3RefModsAxis         — A/B mod pairs on one signed slider each (negative -> A, positive -> B)
  H3RefModApply         — inject the bundle into a MINIMAX_H3_COND conditioning or the
                          built-in ComfyUI CONDITIONING (one node, old ApplyCond
                          workflows auto-migrate via node replacement)

Mod-creation nodes (H3RefModExtract / H3RefModCreateFromFolder) live in
``create_refmod.py`` — a single location for both creation methods.

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

import torch

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
    H3RefMod,
    _blur_latent,
    curve_value_at,
    read_refmod_meta,
)
from ..py.debug_grid import (
    graph_pnginfo,
    pil_to_tensor,
    read_graph_meta,
    render_debug_grid,
)

_PACK_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_MODS_DIR = os.path.join(_PACK_DIR, "mods")  # pre-models/refmods storage, still read
_MOD_CACHE: Dict[str, H3RefMod] = {}
_MOD_CACHE_MAX = 24          # cap: never pin more mods in RAM than this (FIFO eviction)
_MOD_LIST_CACHE_KEY = None   # (dirs, mtimes, sizes) signature of the last _list_mod_names() scan
_MOD_LIST_CACHE_VAL = None
_MOD_SKIP_DIRS = {"graph_presets", ".git", "__pycache__"}

# Mod storage lives in ComfyUI's models/ tree (created on first run) and is
# registered as a first-class folder type so it shows up next to loras/unet.
try:
    folder_paths.add_model_folder_path("refmods", refmods_dir())
except Exception:
    pass

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

def _mod_search_dirs() -> List[str]:
    dirs = [refmods_dir()]
    root_models_mods = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "mods")
    for d in (root_models_mods, LEGACY_MODS_DIR):
        if d not in dirs and os.path.isdir(d):
            dirs.append(d)
    return dirs


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
    sig = []
    for d in _mod_search_dirs():
        if not os.path.isdir(d):
            continue
        for rel_stem, abs_stem in _iter_mod_paths(d):
            try:
                st = os.stat(abs_stem + ".safetensors")
                sig.append(f"{d}:{rel_stem}:{st.st_size}:{int(st.st_mtime)}")
            except OSError:
                pass
    key = "\n".join(sig)
    if key == _MOD_LIST_CACHE_KEY:
        return _MOD_LIST_CACHE_VAL
    names = set()
    for d in _mod_search_dirs():
        if not os.path.isdir(d):
            continue
        for rel_stem, abs_stem in _iter_mod_paths(d):
            meta = read_refmod_meta(abs_stem)
            if meta is not None and meta.get("kind") in ("image", "video", "audio"):
                names.add(rel_stem)
    _MOD_LIST_CACHE_KEY, _MOD_LIST_CACHE_VAL = key, sorted(names)
    return _MOD_LIST_CACHE_VAL


def _find_mod_path(name: str) -> str:
    for d in _mod_search_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p + ".safetensors"):
            return p
    raise FileNotFoundError(
        f"RefMod '{name}' not found. Searched:\n" +
        "\n".join(f"  - {d}/{name}.safetensors" for d in _mod_search_dirs()))


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


def _resolve_folder(folder: str) -> str:
    """Resolve a folder input: absolute path, a name inside input/, or input/ itself."""
    folder = (folder or "").strip().strip('"')
    if not folder:
        return folder_paths.get_input_directory()
    if os.path.isabs(folder):
        resolved = os.path.normpath(folder)
    else:
        resolved = os.path.join(folder_paths.get_input_directory(), folder)
    if not os.path.isdir(resolved):
        raise ValueError(
            f"folder not found: {folder!r} (looked at '{resolved}'; use an "
            "absolute path or a folder name inside input/).")
    return resolved


def _info_lines(mod: H3RefMod) -> List[str]:
    opt = mod.optimize_steps
    if mod.mode == "encode":
        opt = f"n/a ({mod.optimize_steps} — encode mode stores the actual encode)"
    return [
        "=" * 52,
        f"  MiniMax H3 RefMod: {mod.name}",
        f"  {'concept_type':<18} {mod.concept_type}",
        f"  {'audio_concept_type':<18} {mod.audio_concept_type or '-'}",
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


def _prompt_hint(loads) -> str:
    """Merge loaded mods' concept_type + description into one prompt-ready string.

    e.g. "identity: ginger woman, tattooed neck, black lipstick; pose_motion:
    slow twirl into camera, hair whipping". Concat this onto your positive
    prompt (a string-concat node ahead of CLIP Text Encode) instead of
    retyping each mod's description by hand. Mods with no description are
    skipped — a bare concept_type with nothing to say isn't a useful clue.
    """
    parts = []
    for mod, _strength in loads:
        desc = mod.description
        if desc:
            parts.append(f"{mod.concept_type}: {desc}")
    return "; ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Node: H3RefModsAxis (signed A/B sliders)
# ═══════════════════════════════════════════════════════════════════════════

class H3RefModsAxis:
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

class H3RefModStepCurve:
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
            "minimax_h3_refmod_step_curve",
            _make_step_wrapper((curve_direction, curve_shape, curve_value)))
        print(f"[H3RefModStepCurve] {curve_direction} + {curve_shape} "
              f"@ {curve_value:.2f} attached — refs re-mixed per denoising step")
        return (model,)


# ═══════════════════════════════════════════════════════════════════════════
# Node: H3RefModFolderLoader
# ═══════════════════════════════════════════════════════════════════════════

class H3RefModFolderLoader:
    """Load every image/video in a folder as an ordered ref list.

    Feed the ``refs_bundle`` input of Extract H3 RefMod to bulk-extract a
    whole folder (e.g. all photos of a character).  Images load first (by
    filename), then videos; unreadable files are skipped with a note.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": "",
                    "tooltip": "Folder with reference images/videos. An absolute path, or a folder "
                               "name inside ComfyUI's input/ directory (empty = input/ itself)."}),
                "max_items": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1,
                    "display": "number",
                    "tooltip": "Max media files loaded (images first, then videos, by filename)."}),
                "max_frames": ("INT", {"default": 240, "min": 2, "max": 4800, "step": 1,
                    "display": "number",
                    "tooltip": "Video frames kept (uniformly sampled during decode, so a long video "
                               "is never fully decoded into RAM — memory stays bounded by this cap "
                               "x max_edge resolution). 240 = ~10s at 24fps."}),
                "max_edge": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64,
                    "display": "number",
                    "tooltip": "Longest edge in px for loaded images/videos (downscale only, never "
                               "upscale). Loading many 4K files at native resolution is what OOMs "
                               "ComfyUI — the Extract node resizes to ref_resolution anyway, so "
                               "1024-1280 is plenty for folder extraction."}),
            },
        }

    RETURN_TYPES = ("H3_REF_LIST", "INT")
    RETURN_NAMES = ("refs", "count")
    FUNCTION = "load"
    CATEGORY = "H3RefMod"

    @classmethod
    def VALIDATE_INPUTS(cls, folder):
        try:
            _resolve_folder(folder)
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def IS_CHANGED(cls, folder, max_items=32, max_frames=240, max_edge=1024):
        try:
            images, videos = list_media_files(_resolve_folder(folder))
            parts = []
            for p in (images + videos)[:max_items]:
                try:
                    st = os.stat(p)
                    parts.append(f"{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}")
                except OSError:
                    parts.append(f"{os.path.basename(p)}:missing")
            return "|".join(parts)
        except Exception:
            return ""

    def load(self, folder, max_items=32, max_frames=240, max_edge=1024):
        folder = _resolve_folder(folder)
        images, videos = list_media_files(folder)
        items = (images + videos)[:max_items]
        total = len(items)
        refs, failed = [], []
        pbar = comfy.utils.ProgressBar(total)
        for i, p in enumerate(items, start=1):
            kind = "video" if p in videos else "image"
            print(f"[H3RefModFolderLoader] [{i}/{total}] loading {kind} "
                  f"{os.path.basename(p)}")
            try:
                if p in images:
                    refs.append(load_image_file(p, max_edge=max_edge))
                else:
                    refs.append(load_video_file(p, max_frames=max_frames, max_edge=max_edge))
            except Exception as exc:
                failed.append(f"{os.path.basename(p)} ({type(exc).__name__})")
                pbar.update_absolute(i)
                continue
            print(f"[H3RefModFolderLoader] [{i}/{total}] {os.path.basename(p)} "
                  f"-> {tuple(refs[-1].shape)}")
            pbar.update_absolute(i)
        if failed:
            print(f"[H3RefModFolderLoader] skipped unreadable files: {', '.join(failed)}")
        if not refs:
            raise ValueError(
                f"H3RefModFolderLoader: no images/videos found in {folder} "
                "(images: png/jpg/jpeg/webp/bmp/gif, videos: mp4/webm/mov/mkv/avi/m4v).")
        n_vid = sum(r.shape[0] > 1 for r in refs)
        print(f"[H3RefModFolderLoader] loaded {len(refs)} media from {folder} "
              f"({n_vid} video, {len(refs) - n_vid} image)")
        return (refs, len(refs))


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "H3RefModFolderLoader": H3RefModFolderLoader,
    "H3RefModsAxis": H3RefModsAxis,
    "H3RefModApplyAdvanced": H3RefModApplyAdvanced,
    "H3RefModApplySimple": H3RefModApplySimple,
    "H3RefModStepCurve": H3RefModStepCurve,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModFolderLoader": "Load H3 RefMod Folder",
    "H3RefModsAxis": "Load H3 RefMod Axis",
    "H3RefModApplyAdvanced": "Apply H3 RefMod Advanced",
    "H3RefModApplySimple": "Apply H3 RefMod",
    "H3RefModStepCurve": "H3 RefMod Step Curve",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
