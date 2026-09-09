from __future__ import annotations

import os
from typing import Dict, List, Tuple

import server

from .refmod_apply import _prompt_hint
from ..py.refmod_browser import (
    PREVIEW_EXTS,
    browser_root,
    list_browser_dir,
    safe_file_path,
)
from ..py.refmod_core import H3RefMod


_VISUAL_MOD_CACHE: Dict[str, H3RefMod] = {}
_VISUAL_MOD_CACHE_MAX = 24
VISUAL_SUFFIX = "_Video"
AUDIO_SUFFIX = "_Audio"


def _load_single_mod_from_path(mod_path: str) -> H3RefMod:
    path = safe_file_path(mod_path)
    if not path.endswith(".safetensors"):
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
    if path.endswith(VISUAL_SUFFIX + ".safetensors"):
        audio_path = path[:-len(VISUAL_SUFFIX + ".safetensors")] + AUDIO_SUFFIX + ".safetensors"
        if os.path.isfile(audio_path):
            mods.append(_load_single_mod_from_path(audio_path))
    return mods


class H3RefModVisualPicker:
    """Pick a RefMod by browsing models/refmods thumbnails and append it to a bundle."""

    NAME = "Visual RefMod Picker"
    CATEGORY = "H3RefMod"

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
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "How strongly the visual RefMod reference is preserved.",
                    },
                ),
                "audio_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "How strongly a paired audio RefMod is preserved. Legacy one-file combined RefMods cannot split this from visual strength.",
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
        "beside a *_Video file, both are loaded together and appended to the bundle, "
        "with separate video/audio strengths. Legacy combined RefMods still use one shared strength."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def pick(self, mod_path: str, strength: float, audio_strength: float = 1.0, mods=None):
        rows: List[Tuple[H3RefMod, float]] = []
        if mods is not None and isinstance(mods, (list, tuple)):
            rows.extend(list(mods))

        selected = str(mod_path or "").strip()
        if not selected or selected.lower() == "(none)":
            hint = _prompt_hint(rows)
            return (rows, hint)

        clipped = min(1.0, max(0.0, float(strength)))
        clipped_audio = min(1.0, max(0.0, float(audio_strength)))
        selected_mods = _load_mods_from_path(selected)
        rows.extend(
            (mod, clipped_audio if mod.kind == "audio" else clipped)
            for mod in selected_mods
        )
        print("[H3RefModVisualPicker] " + ", ".join(
            f"{mod.name}@{(clipped_audio if mod.kind == 'audio' else clipped):.2f}" for mod in selected_mods
        ))
        hint = _prompt_hint(rows)
        if hint:
            print(f"[H3RefModVisualPicker] prompt_hint: {hint}")
        return (rows, hint)


NODE_CLASS_MAPPINGS = {"H3RefModVisualPicker": H3RefModVisualPicker}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RefModVisualPicker": "Visual RefMod Picker"}


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