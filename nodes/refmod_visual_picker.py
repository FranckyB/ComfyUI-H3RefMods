from __future__ import annotations

import os
from typing import Dict, List, Tuple

import server

from ..py.refmod_browser import (
    PREVIEW_EXTS,
    browser_root,
    list_browser_dir,
    paired_mod_path,
    safe_file_path,
)
from ..py.refmod_core import H3RefMod
from ..py.refmod_common import _prompt_hint
from .refmod_loader import _MAX_WEIGHT, _append_weighted_mod, _weight_display

_VISUAL_MOD_CACHE: Dict[str, H3RefMod] = {}
_VISUAL_MOD_CACHE_MAX = 24


def _load_single_mod_from_path(mod_path: str) -> H3RefMod:
    path = safe_file_path(mod_path)
    if not path.lower().endswith(".safetensors"):
        raise ValueError("Selected file is not a RefMod safetensors file.")
    if path in _VISUAL_MOD_CACHE:
        return _VISUAL_MOD_CACHE[path]
    mod = H3RefMod.load(path[:-len(".safetensors")], device="cpu")
    _VISUAL_MOD_CACHE[path] = mod
    if len(_VISUAL_MOD_CACHE) > _VISUAL_MOD_CACHE_MAX:
        _VISUAL_MOD_CACHE.pop(next(iter(_VISUAL_MOD_CACHE)))
    return mod


def _load_mods_from_path(mod_path: str) -> List[H3RefMod]:
    path = safe_file_path(mod_path)
    mods = [_load_single_mod_from_path(path)]
    paired_audio = paired_mod_path(path[:-len(".safetensors")], "audio")
    if paired_audio is not None:
        mods.append(_load_single_mod_from_path(paired_audio))
    return mods


class H3RefModVisualPicker:
    """Pick a RefMod by browsing models/refmods thumbnails and append it to a bundle."""

    NAME = "Visual RefMod Picker"
    CATEGORY = "H3RefModPicker"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mod_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Selected RefMod safetensors path from the thumbnail browser.",
                    },
                ),
                "video_weight": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": _MAX_WEIGHT,
                        "step": 0.01,
                        "tooltip": "Unified video weight control. 0 skips video. 0..1 behaves like the old strength control. Values above 1 repeat the same video RefMod as extra copies.",
                    },
                ),
                "audio_weight": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": _MAX_WEIGHT,
                        "step": 0.01,
                        "tooltip": "Unified audio weight control. 0 skips audio. 0..1 behaves like the old audio strength control. Values above 1 repeat the same audio RefMod as extra copies.",
                    },
                ),
            },
            "optional": {
                "mods": ("H3_REF_MODS",),
            },
        }

    RETURN_TYPES = ("H3_REF_MODS", "STRING")
    RETURN_NAMES = ("mods", "prompt_hint")
    FUNCTION = "pick"
    DESCRIPTION = (
        "Pick a RefMod by browsing models/refmods and its subfolders. "
        "A matching preview image with the same base name is shown when present; "
        "otherwise a placeholder is used. When a matching *_Audio file exists "
        "beside a *_Video or *_Visual file, both are loaded together and appended to the bundle, "
        "with separate video/audio weights. A weight in 0..1 behaves like the old strength control; "
        "a weight above 1 repeats the same RefMod as extra copies. Legacy combined RefMods still use one shared video weight."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def pick(self, mod_path: str, video_weight: float = 1.0, audio_weight: float = 1.0,
             mods=None, strength=None, audio_strength=None):
        rows: List[Tuple[H3RefMod, float]] = []
        if mods is not None and isinstance(mods, (list, tuple)):
            rows.extend(list(mods))

        selected = str(mod_path or "").strip()
        if not selected or selected.lower() == "(none)":
            hint = _prompt_hint(rows)
            return (rows, hint)

        if strength is not None:
            video_weight = strength
        if audio_strength is not None:
            audio_weight = audio_strength
        selected_mods = _load_mods_from_path(selected)
        hint_rows = list(mods) if mods is not None and isinstance(mods, (list, tuple)) else []
        summaries = []
        for mod in selected_mods:
            weight = audio_weight if mod.kind == "audio" else video_weight
            clipped = _append_weighted_mod(rows, mod, weight)
            if clipped > 0.0:
                hint_rows.append((mod, min(1.0, clipped)))
            summaries.append(f"{mod.name}({_weight_display(clipped)})")
        print("[H3RefModVisualPicker] " + ", ".join(summaries))
        hint = _prompt_hint(hint_rows)
        if hint:
            print(f"[H3RefModVisualPicker] prompt_hint: {hint}")
        return (rows, hint)


NODE_CLASS_MAPPINGS = {"H3RefModVisualPicker": H3RefModVisualPicker}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RefModVisualPicker": "Visual H3 RefMod Picker"}


@server.PromptServer.instance.routes.get("/h3refmods/refmod-browser/root")
async def refmod_browser_root(request):
    try:
        return server.web.json_response({"ok": True, "root": browser_root()})
    except Exception as exc:
        return server.web.json_response({"ok": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.get("/h3refmods/refmod-browser/list")
async def refmod_browser_list(request):
    try:
        data = list_browser_dir(request.query.get("path", "").strip())
        return server.web.json_response({"ok": True, **data})
    except ValueError as exc:
        return server.web.json_response({"ok": False, "error": str(exc)}, status=404)
    except Exception as exc:
        return server.web.json_response({"ok": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.get("/h3refmods/refmod-browser/file")
async def refmod_browser_file(request):
    try:
        path = safe_file_path(request.query.get("path", "").strip())
        if os.path.splitext(path)[1].lower() not in PREVIEW_EXTS:
            return server.web.json_response(
                {"ok": False, "error": "Unsupported file type"}, status=403
            )
        return server.web.FileResponse(path, headers={"Cache-Control": "no-cache"})
    except ValueError as exc:
        return server.web.json_response({"ok": False, "error": str(exc)}, status=404)
    except Exception as exc:
        return server.web.json_response({"ok": False, "error": str(exc)}, status=500)