"""refmod_loader.py — RefMod loader nodes.

Loader surfaces now emit only ``H3_REF_MODS`` bundles so every downstream
consumer sees one consistent row shape: plain ``(mod, strength)`` tuples.

    H3RefModLoader   — load one RefMod and optionally append it to an existing
                                         bundle, visual-picker style.
    H3RefModStacker  — load 1-8 RefMods from combo slots into one bundle.
"""

from __future__ import annotations

from typing import Any, List

from comfy_api.latest import io

from .refmod_apply import _list_mod_names, _load_mod

from ..py.refmod_core import H3RefMod


class H3RefModLoader:
    """Load one RefMod and append it to an existing bundle."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mod": (_list_mod_names(), {"tooltip": "The RefMod to load."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "number",
                    "tooltip": "How strongly this mod's reference is preserved. 1.0 = full ref "
                               "(official behavior). Lower values blur the ref toward a softened "
                               "copy of itself — identity fades smoothly instead of turning into "
                               "static/noise texture. 0 skips the mod entirely."}),
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

    def load(self, mod, strength=1.0, mods=None):
        rows = list(mods) if mods is not None else []
        m = _load_mod(mod)
        strength = min(1.0, max(0.0, float(strength)))
        rows.append((m, strength))
        print(f"[H3RefModLoader] {m.name}@{strength:.2f}")
        hint = _prompt_hint(rows)
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
            required[f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                "step": 0.01, "display": "number",
                "tooltip": "How strongly this mod's reference is preserved. 1.0 = full ref (official "
                           "behavior). Lower values blur the ref toward a softened copy of itself — "
                           "identity fades smoothly and stays plausible instead of turning into "
                           "static/noise texture. 0 skips the mod entirely."})
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
        for i in range(1, self.MAX_SLOTS + 1):
            name = str(kwargs.get(f"mod_{i}", self.NONE))
            strength = float(kwargs.get(f"strength_{i}", 1.0))
            if not name or name == self.NONE or strength <= 0.0:
                continue
            rows.append((_load_mod(name), min(1.0, max(0.0, strength))))
        if rows:
            print("[H3RefModStacker] " + ", ".join(
                f"{m.name}@{s:.2f}"
                for m, s in rows)
                + f" ({sum(m.token_count for m, _s in rows)} tokens total)")
        else:
            print("[H3RefModStacker] no mods selected "
                  "(all slots (none) or strength 0)")
        if show_info:
            for mod, strength in rows:
                print("\n".join(_info_lines(mod)))
                print(f"  {'strength':<18} {strength:.2f}")
        hint = _prompt_hint(rows)
        if hint:
            print(f"[H3RefModStacker] prompt_hint: {hint}")
        return (rows, hint)


def _prompt_hint(rows):
    """Build a merged prompt hint from the RefMod descriptions in a bundle."""
    parts = []
    for mod, _strength in rows:
        description = getattr(mod, "description", "")
        if not description:
            continue
        if isinstance(description, str):
            cleaned = description.strip()
        else:
            cleaned = str(description).strip()
        if cleaned:
            parts.append(f"{mod.concept_type}: {cleaned}")
    return "; ".join(parts)

def _info_lines(mod: H3RefMod) -> List[str]:
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

NODE_CLASS_MAPPINGS = {
    "H3RefModLoader":   H3RefModLoader,
    "H3RefModStacker":  H3RefModStacker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModLoader":   "Load RefMod",
    "H3RefModStacker":  "Load RefMod Stack",
}
