"""refmod_loader.py — RefMod loader nodes.

Loader surfaces now emit only ``H3_REF_MODS`` bundles so every downstream
consumer sees one consistent row shape: plain ``(mod, strength)`` tuples.

    H3RefModLoader   — load one RefMod and optionally append it to an existing
                                         bundle, visual-picker style.
    H3RefModStacker  — load 1-8 RefMods from combo slots into one bundle.
"""

from __future__ import annotations

from typing import Any, Dict, List
import os
import folder_paths
import comfy.utils

from ..py.refmod_core import H3RefMod, read_refmod_meta

from ..py.refmod_common import (
    list_media_files,
    load_image_file,
    load_video_file,
    _prompt_hint,
    _info_lines,
)


from ..py.refmod_common import (
    refmods_dir,
)

_MOD_CACHE: Dict[str, H3RefMod] = {}
_MOD_CACHE_MAX = 24                                # cap: never pin more mods in RAM than this (FIFO eviction)
_MOD_LIST_CACHE_KEY = None                         # (dirs, mtimes, sizes) signature of the last _list_mod_names() scan
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


class H3RefModStacker:
    """Stack 1-8 RefMods in one node, each with its own strength override."""

    MAX_SLOTS = 8
    NONE = "(none)"

    @classmethod
    def INPUT_TYPES(cls):
        names = [cls.NONE] + _list_mod_names()
        required: dict[str, tuple[Any, dict[str, Any]]] = {
            "show_info": ("BOOLEAN", {"default": False,
                "tooltip": "Print full details (tokens, layout, source, pool) of every loaded mod to the console."}),
        }
        for i in range(1, cls.MAX_SLOTS + 1):
            required[f"mod_{i}"] = (names, {"tooltip": f"RefMod {i} to load, or {cls.NONE}."})
            required[f"weight_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": _MAX_WEIGHT,
                "step": 0.01, "display": "number",
                "tooltip": "Unified weight control. 0 skips the mod. 0..1 behaves like the old "
                           "strength control. Values above 1 repeat the same RefMod as extra copies: "
                           "for example 2.7 becomes two full copies plus one 0.7 copy."})
        return {"required": required}

    RETURN_TYPES = ("H3_REF_MODS", "STRING")
    RETURN_NAMES = ("mods", "prompt_hint")
    FUNCTION = "stack"
    CATEGORY = "H3RefMod"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        available = set(_list_mod_names())
        for i in range(1, cls.MAX_SLOTS + 1):
            name = str(kwargs.get(f"mod_{i}", cls.NONE))
            if name and name != cls.NONE and name not in available:
                return (f"RefMod slot {i}: '{name}' not found in mods/. "
                        "Run Extract H3 RefMod first.")
        return True

    def stack(self, show_info=False, **kwargs):
        rows = []  # (mod, strength)
        info_rows = []  # (mod, weight)
        for i in range(1, self.MAX_SLOTS + 1):
            name = str(kwargs.get(f"mod_{i}", self.NONE))
            weight = float(kwargs.get(f"weight_{i}", kwargs.get(f"strength_{i}", 1.0)))
            if not name or name == self.NONE or weight <= 0.0:
                continue
            mod = _load_mod(name)
            clipped = _append_weighted_mod(rows, mod, weight)
            info_rows.append((mod, clipped))
        if rows:
            print("[H3RefModStacker] " + ", ".join(
                f"{m.name}({_weight_display(weight)})"
                for m, weight in info_rows)
                + f" ({sum(m.token_count for m, _s in rows)} tokens total)")
        else:
            print("[H3RefModStacker] no mods selected "
                  "(all slots (none) or weight 0)")
        if show_info:
            for mod, weight in info_rows:
                print("\n".join(_info_lines(mod)))
                print(f"  {'weight':<18} {weight:.2f}")
        hint = _prompt_hint([(mod, min(1.0, weight)) for mod, weight in info_rows])
        if hint:
            print(f"[H3RefModStacker] prompt_hint: {hint}")
        return (rows, hint)


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


NODE_CLASS_MAPPINGS = {
    "H3RefModLoader":       H3RefModLoader,
    "H3RefModStacker":      H3RefModStacker,
    "H3RefModFolderLoader": H3RefModFolderLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModLoader":       "Load H3 RefMod",
    "H3RefModStacker":      "Load H3 RefMod Stack",
    "H3RefModFolderLoader": "Load H3 RefMod Folder",
}
