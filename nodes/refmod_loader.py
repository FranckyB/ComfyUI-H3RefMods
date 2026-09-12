"""
H3RefModLoader — load one RefMod and optionally append it to another
"""

from __future__ import annotations

from typing import Any, Dict, List
import os
import folder_paths

from ..py.refmod_core import H3RefMod, read_refmod_meta

from ..py.refmod_common import (
    _prompt_hint,
)


from ..py.refmod_common import (
    refmods_dir,
)

_MOD_CACHE: Dict[str, H3RefMod] = {}
_MOD_CACHE_MAX = 24            # cap: never pin more mods in RAM than this (FIFO eviction)
_MOD_LIST_CACHE_KEY = None     # (dirs, mtimes, sizes) signature of the last _list_mod_names() scan
_MOD_LIST_CACHE_VAL = None
_MOD_SKIP_DIRS = {"graph_presets", ".git", "__pycache__"}
_MAX_WEIGHT = 10.0

# Mod storage lives in ComfyUI's models/ tree (created on first run) and is
# registered as a first-class folder type so it shows up next to loras/unet.
try:
    folder_paths.add_model_folder_path("refmods", refmods_dir())
except Exception:
    pass

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
        _MOD_CACHE.pop(next(iter(_MOD_CACHE)))
    return mod


def _normalize_weight(weight: float) -> float:
    return min(_MAX_WEIGHT, max(0.0, float(weight)))


def _expand_weight(weight: float) -> List[float]:
    clipped = _normalize_weight(weight)
    whole = int(clipped)
    remainder = clipped - whole
    strengths = [1.0] * whole
    if remainder > 1e-6:
        strengths.append(remainder)
    return strengths


def _append_weighted_mod(rows: List[tuple[H3RefMod, float]], mod: H3RefMod, weight: float) -> float:
    clipped = _normalize_weight(weight)
    rows.extend((mod, strength) for strength in _expand_weight(clipped))
    return clipped


def _weight_display(weight: float) -> str:
    clipped = _normalize_weight(weight)
    whole = int(clipped)
    remainder = clipped - whole
    if whole <= 0:
        return f"{clipped:.2f}"
    if remainder <= 1e-6:
        return f"x{whole}"
    return f"x{whole} + {remainder:.2f}"

class H3RefModLoader:
    """Load one RefMod and append it to an existing bundle."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mod": (_list_mod_names(), {"tooltip": "The RefMod to load."}),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": _MAX_WEIGHT, "step": 0.01,
                    "display": "number",
                    "tooltip": "Unified weight control. 0 skips the mod. 0..1 behaves like the old "
                               "strength control. Values above 1 repeat the same RefMod as extra "
                               "copies: for example 2.7 becomes two full copies plus one 0.7 copy."}),
            },
            "optional": {
                "mods": ("H3_REF_MODS",),
            },
        }

    RETURN_TYPES = ("H3_REF_MODS", "STRING")
    RETURN_NAMES = ("mods", "prompt_hint")
    FUNCTION = "load"
    CATEGORY = "H3RefMod"

    @classmethod
    def VALIDATE_INPUTS(cls, mod, **kwargs):
        if mod not in set(_list_mod_names()):
            return f"RefMod '{mod}' not found in mods/. Run Extract H3 RefMod first."
        return True

    def load(self, mod, weight=1.0, mods=None, strength=None):
        rows = list(mods) if mods is not None else []
        m = _load_mod(mod)
        if strength is not None:
            weight = strength
        clipped = _append_weighted_mod(rows, m, weight)
        print(f"[H3RefModLoader] {m.name} weight={clipped:.2f} -> {_weight_display(clipped)}")
        hint_rows = list(mods) if mods is not None else []
        if clipped > 0.0:
            hint_rows.append((m, min(1.0, clipped)))
        hint = _prompt_hint(hint_rows)
        if hint:
            print(f"[H3RefModLoader] prompt_hint: {hint}")
        return (rows, hint)


NODE_CLASS_MAPPINGS = {
    "H3RefModLoader": H3RefModLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModLoader": "Load H3 RefMod"
}
