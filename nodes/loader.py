"""
loader.py — H3RefModsLoader node.

Load 1-8 RefMods in one node, each with its own typed strength and a per-row
``copies`` boost (LoRA-loader style).  Outputs an ``H3_REF_MODS`` bundle for
Apply H3 RefMod, plus a ``prompt_hint`` string merging each mod's
concept_type + description (concat it onto your prompt instead of retyping).
"""

from __future__ import annotations

from .nodes import (
    _info_lines,
    _list_mod_names,
    _load_mod,
    _prompt_hint,
)


class H3RefModsLoader:
    """Load 1-8 RefMods in one node, each with its own typed strength."""

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
            required[f"copies_{i}"] = ("INT", {"default": 1, "min": 1, "max": 10, "step": 1,
                "display": "number",
                "tooltip": "How many copies of this mod to inject (1 = normal, 2+ = the same ref "
                           "repeated — the manual row-duplication trick as a knob, up to 10x). More "
                           "copies = noticeably stronger reference, but each copy costs its full "
                           "token count in every DiT block, so it slows down inference and eats "
                           "VRAM — 2-3 copies is the sweet spot, 10x will be very slow."})
        return {"required": required}

    RETURN_TYPES = ("H3_REF_MODS", "STRING")
    RETURN_NAMES = ("mods", "prompt_hint")
    FUNCTION = "load"
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

    def load(self, show_info=False, **kwargs):
        rows = []  # (mod, strength, copies)
        for i in range(1, self.MAX_SLOTS + 1):
            name = str(kwargs.get(f"mod_{i}", self.NONE))
            strength = float(kwargs.get(f"strength_{i}", 1.0))
            if not name or name == self.NONE or strength <= 0.0:
                continue
            rows.append((_load_mod(name), min(1.0, max(0.0, strength)),
                         int(kwargs.get(f"copies_{i}", 1))))
        loads = []
        for mod, strength, copies in rows:
            loads.extend([(mod, strength)] * copies)
        if loads:
            print("[H3RefModsLoader] " + ", ".join(
                f"{m.name}@{s:.2f}" + (f" x{c}" if c > 1 else "")
                for m, s, c in rows)
                + f" ({sum(m.token_count * c for m, _, c in rows)} tokens total)")
        else:
            print("[H3RefModsLoader] no mods selected "
                  "(all slots (none) or strength 0)")
        if show_info:
            for mod, strength, copies in rows:
                print("\n".join(_info_lines(mod)))
                print(f"  {'strength':<18} {strength:.2f}"
                      + (f"  (x{copies} copies)" if copies > 1 else ""))
        hint = _prompt_hint([(m, s) for m, s, _ in rows])
        if hint:
            print(f"[H3RefModsLoader] prompt_hint: {hint}")
        return (loads, hint)


NODE_CLASS_MAPPINGS = {
    "H3RefModsLoader": H3RefModsLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModsLoader": "Load H3 RefMods",
}
