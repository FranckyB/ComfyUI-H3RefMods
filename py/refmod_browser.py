from __future__ import annotations

import os
from typing import Dict, List, Optional

from .refmod_common import refmods_dir
from .refmod_core import read_refmod_meta


PREVIEW_EXTS = (".png", ".jpg", ".jpeg", ".webp")
HIDDEN_BROWSER_DIRS = {"graph_presets"}
VISUAL_SUFFIXES = ("_Video", "_Visual")
AUDIO_SUFFIXES = ("_Audio",)


def browser_root() -> str:
    root = os.path.abspath(refmods_dir())
    os.makedirs(root, exist_ok=True)
    return root


def _safe_abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _is_under_root(path: str) -> bool:
    root = os.path.realpath(browser_root())
    current = os.path.realpath(_safe_abspath(path))
    return current == root or current.startswith(root + os.sep)


def safe_dir_path(path: str = "") -> str:
    if not path:
        return browser_root()
    current = _safe_abspath(path)
    if not _is_under_root(current):
        raise ValueError("Path is outside models/refmods")
    if not os.path.isdir(current):
        raise ValueError("Folder not found")
    return current


def safe_file_path(path: str) -> str:
    if not path:
        raise ValueError("Missing path")
    current = _safe_abspath(path)
    if not _is_under_root(current):
        raise ValueError("Path is outside models/refmods")
    if not os.path.isfile(current):
        raise ValueError("File not found")
    return current


def _strip_known_suffix(path_no_ext: str) -> tuple[str, str | None, str | None]:
    name = os.path.basename(path_no_ext)
    lower_name = name.lower()
    for suffix in VISUAL_SUFFIXES:
        if lower_name.endswith(suffix.lower()):
            return (path_no_ext[:-len(suffix)], "visual", suffix)
    for suffix in AUDIO_SUFFIXES:
        if lower_name.endswith(suffix.lower()):
            return (path_no_ext[:-len(suffix)], "audio", suffix)
    return (path_no_ext, None, None)


def paired_mod_path(path_no_ext: str, target_kind: str) -> Optional[str]:
    base, current_kind, _suffix = _strip_known_suffix(path_no_ext)
    if current_kind is None:
        return None
    suffixes = VISUAL_SUFFIXES if target_kind == "visual" else AUDIO_SUFFIXES
    for suffix in suffixes:
        candidate = base + suffix + ".safetensors"
        if os.path.isfile(candidate):
            return candidate
    return None


def _find_preview(path_no_ext: str) -> Optional[str]:
    candidates = [path_no_ext]
    base, kind, _suffix = _strip_known_suffix(path_no_ext)
    if kind is not None:
        candidates.insert(0, base)
    for stem in candidates:
        for ext in PREVIEW_EXTS:
            candidate = stem + ext
            if os.path.isfile(candidate):
                return candidate
    return None


def _display_mod_name(path_no_ext: str) -> str:
    base, kind, _suffix = _strip_known_suffix(path_no_ext)
    if kind is not None:
        name = os.path.basename(base)
        return name or os.path.basename(path_no_ext)
    return os.path.basename(path_no_ext)


def _mod_entry(path: str) -> Optional[Dict]:
    if not path.lower().endswith(".safetensors"):
        return None
    path_no_ext = path[:-len(".safetensors")]
    meta = read_refmod_meta(path_no_ext)
    if meta is None or meta.get("kind") not in ("image", "video", "audio"):
        return None
    _base, kind, _suffix = _strip_known_suffix(path_no_ext)
    if kind == "audio" and paired_mod_path(path_no_ext, "visual"):
        return None
    preview = _find_preview(path_no_ext)
    return {
        "name": _display_mod_name(path_no_ext) + ".safetensors",
        "path": path,
        "preview_path": preview,
        "concept_type": str(meta.get("concept_type", "generic") or "generic"),
        "description": str(meta.get("description", "") or ""),
    }


def list_browser_dir(path: str = "") -> Dict:
    current = safe_dir_path(path)
    dirs: List[Dict] = []
    mods: List[Dict] = []

    try:
        entries = list(os.scandir(current))
    except PermissionError as exc:
        raise ValueError("Access denied") from exc

    for entry in entries:
        name = entry.name
        if name.startswith(".") or name in HIDDEN_BROWSER_DIRS:
            continue
        try:
            if entry.is_dir(follow_symlinks=False):
                dirs.append({"name": name, "path": os.path.abspath(entry.path)})
            elif entry.is_file(follow_symlinks=False) and name.lower().endswith(".safetensors"):
                item = _mod_entry(os.path.abspath(entry.path))
                if item is not None:
                    mods.append(item)
        except OSError:
            continue

    dirs.sort(key=lambda item: item["name"].lower())
    mods.sort(key=lambda item: item["name"].lower())

    parent = os.path.dirname(current.rstrip("\\/"))
    parent_path = parent if parent and _is_under_root(parent) and parent != current else None
    return {
        "root": browser_root(),
        "current_path": current,
        "parent_path": parent_path,
        "dirs": dirs,
        "mods": mods,
    }