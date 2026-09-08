"""
refmod_single.py — H3RefModSingleLoader + H3RefModsCombine.

A LoRA-loader-style pair, as an alternative to the fixed-size
H3RefModStacker (nodes/refmod_stacker.py):

  H3RefModLoader — load ONE RefMod with its own strength and an
                          optional description override, output as a single
                          ``H3_REFMOD`` row.
  H3RefModsCombine      — collect ``H3_REFMOD`` rows into one ``H3_REF_MODS``
                          bundle; the input grows a new slot every time you
                          connect another Load H3 RefMod node (autogrow,
                          same "+" pattern as Extract H3 RefMod's ref_image_N/
                          ref_video_N inputs), instead of picking from a
                          fixed 8-row combo list.

Why a description override at load time instead of baked into the mod
───────────────────────────────────────────────────────────────────────────
A mod's ``description`` (set at Extract time) is what ``prompt_hint`` merges
into a prompt string (concept_type + description, e.g. "identity: a ginger
woman with tattoos"). The same character/motion mod often needs a different
description depending on what you're generating this time (a different
outfit, a different action) — baking one fixed description into the saved
``.safetensors`` means re-extracting just to reword it.
``H3RefModLoader``'s ``description`` widget overrides the mod's stored
description for this workflow only; leave it empty to keep using what was
saved at Extract time.

The resulting bundle (a list of ``(mod, strength, description_override)``
rows) is the same ``H3_REF_MODS`` type as H3RefModStacker/H3RefModsAxis
produce (those still emit plain ``(mod, strength)`` 2-tuples) — every
consumer (Apply H3 RefMod, MiniMax H3 RefMods to Video) goes through
``nodes.py``'s ``_unpack_row`` helper, so bundles from either loader style
work interchangeably and can even be mixed.
"""

from __future__ import annotations

from typing import Any, List

from comfy_api.latest import io

from .refmod_apply import _list_mod_names, _load_mod

from ..py.refmod_core import H3RefMod


class H3RefModLoader:
    """Load a single RefMod with its own strength and an optional description override."""

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
                "description": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Override this mod's stored description for this workflow (leave "
                               "empty to use the description saved at Extract time). Merged into "
                               "prompt_hint. Doesn't touch the saved mod file or its reference "
                               "latent."}),
            },
        }

    RETURN_TYPES = ("H3_REFMOD",)
    RETURN_NAMES = ("mod",)
    FUNCTION = "load"
    CATEGORY = "H3RefMod"

    @classmethod
    def VALIDATE_INPUTS(cls, mod, **kwargs):
        if mod not in set(_list_mod_names()):
            return f"RefMod '{mod}' not found in mods/. Run Extract H3 RefMod first."
        return True

    def load(self, mod, strength=1.0, description=""):
        m = _load_mod(mod)
        strength = min(1.0, max(0.0, float(strength)))
        desc = description.strip() or None
        print(f"[H3RefModLoader] {m.name}@{strength:.2f}"
              + (f" (description override: {desc!r})" if desc else ""))
        return ((m, strength, desc),)


class H3RefModStacker:
    """Stack 1-8 RefMods in one node, each with its own strength and description override."""

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


def _prompt_hint(rows):
    """Build a merged prompt hint from the RefMod descriptions in a bundle."""
    parts = []
    for mod, _strength, desc_override in rows:
        # Use the workflow override when supplied; otherwise keep the description saved in the mod.
        description = desc_override or getattr(mod, "description", "")
        if not description:
            continue
        if isinstance(description, str):
            cleaned = description.strip()
        else:
            cleaned = str(description).strip()
        if cleaned:
            parts.append(cleaned)
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

class H3RefModsCombine(io.ComfyNode):
    """Combine individual 'Load H3 RefMod' outputs into one H3_REF_MODS bundle.

    Connect Load H3 RefMod (single) nodes here — the "+" button on ``mods``
    grows a new slot each time, LoRA-stack style, instead of a fixed-size
    loader.  Connection order = subject/slot order downstream (MiniMax H3
    RefMods to Video's ``<Subject N>`` numbering, Apply H3 RefMod's ref
    order).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RefModsCombine",
            display_name="Combine H3 RefMods",
            category="H3RefMod",
            description="Combine individual 'Load H3 RefMod' outputs into one H3_REF_MODS "
                        "bundle. The input grows as you connect more mods (LoRA-stack style); "
                        "connection order = subject/slot order downstream.",
            inputs=[
                io.Autogrow.Input("mods", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Custom("H3_REFMOD").Input("mod",
                            tooltip="One RefMod load (from Load H3 RefMod), with its own "
                                    "strength and optional description override."),
                        names=[f"mod_{i}" for i in range(1, 33)], min=0)),
            ],
            outputs=[
                io.Custom("H3_REF_MODS").Output("mods",
                    tooltip="Bundle for Apply H3 RefMod / MiniMax H3 RefMods to Video. Slot "
                            "order = connection order."),
            ],
        )

    @classmethod
    def execute(cls, mods=None) -> io.NodeOutput:
        ordered = []
        for key in sorted((mods or {}).keys(), key=lambda k: int(k.rsplit("_", 1)[1])):
            row = (mods or {})[key]
            if row is not None:
                ordered.append(row)
        if not ordered:
            raise ValueError("H3RefModsCombine: connect at least one Load H3 RefMod.")
        print("[H3RefModsCombine] " + ", ".join(
            f"{m.name}@{s:.2f}" + (f" ({d})" if d else "")
            for m, s, d in ordered)
            + f" ({sum(m.token_count for m, _s, _d in ordered)} tokens total)")
        return io.NodeOutput(ordered)


NODE_CLASS_MAPPINGS = {
    "H3RefModLoader":   H3RefModLoader,
    "H3RefModStacker":  H3RefModStacker,
    "H3RefModsCombine": H3RefModsCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModLoader":   "Load RefMod",
    "H3RefModStacker":  "Load RefMod Stack",
    "H3RefModsCombine": "Combine RefMods",
}
