"""
refmod_stacker.py — H3RefModStacker node.

Stack 1-8 RefMods in one node, each with its own typed strength and an
optional description override (LoRA-stacker style).  Outputs an
``H3_REF_MODS`` bundle for Apply H3 RefMod / MiniMax H3 RefMods to Video,
plus a ``prompt_hint`` string merging each mod's concept_type + description
(concat it onto your prompt instead of retyping).

``description_N`` overrides that mod's stored description for this workflow
only (merged into ``prompt_hint``) — leave empty to use whatever was saved at
Extract time.  It doesn't touch the saved mod file.
"""

from __future__ import annotations

from .nodes import (
    _info_lines,
    _list_mod_names,
    _load_mod,
    _prompt_hint,
)


class H3RefModStacker:
    """Stack 1-8 RefMods in one node, each with its own strength and description override."""

    MAX_SLOTS = 8
    NONE = "(none)"

    @classmethod
    def INPUT_TYPES(cls):
        names = [cls.NONE] + _list_mod_names()
        required = {
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
            required[f"description_{i}"] = ("STRING", {"default": "", "multiline": True,
                "tooltip": f"Override RefMod {i}'s stored description for this workflow (leave "
                           f"empty to use the description saved at Extract time). Merged into "
                           f"prompt_hint. Doesn't touch the saved mod file or its reference "
                           f"latent."})
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
        rows = []  # (mod, strength, description_override)
        for i in range(1, self.MAX_SLOTS + 1):
            name = str(kwargs.get(f"mod_{i}", self.NONE))
            strength = float(kwargs.get(f"strength_{i}", 1.0))
            if not name or name == self.NONE or strength <= 0.0:
                continue
            desc = str(kwargs.get(f"description_{i}", "")).strip() or None
            rows.append((_load_mod(name), min(1.0, max(0.0, strength)), desc))
        if rows:
            print("[H3RefModStacker] " + ", ".join(
                f"{m.name}@{s:.2f}" + (f" ({d})" if d else "")
                for m, s, d in rows)
                + f" ({sum(m.token_count for m, _s, _d in rows)} tokens total)")
        else:
            print("[H3RefModStacker] no mods selected "
                  "(all slots (none) or strength 0)")
        if show_info:
            for mod, strength, desc in rows:
                print("\n".join(_info_lines(mod)))
                print(f"  {'strength':<18} {strength:.2f}"
                      + (f"  (description override: {desc!r})" if desc else ""))
        hint = _prompt_hint(rows)
        if hint:
            print(f"[H3RefModStacker] prompt_hint: {hint}")
        return (rows, hint)


NODE_CLASS_MAPPINGS = {
    "H3RefModStacker": H3RefModStacker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModStacker": "Load RefMod Stack",
}
