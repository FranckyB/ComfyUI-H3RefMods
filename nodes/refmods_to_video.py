"""
refmods_to_video.py — H3RefModsToVideo node.

A RefMod-aware counterpart to ComfyUI's native "MiniMax H3 Reference to Video"
(``comfy_extras.nodes_minimax_h3.MiniMaxH3ReferenceToVideo``).

Design — the RefMod is the ONLY reference
─────────────────────────────────────────
A RefMod's stored latent is the reference, full stop.  This node injects those
latents into the DiT payload (``minimax_refs``) and tokenizes your prompt — it
does NOT re-encode pixels or feed any thumbnail into Qwen's vision encoder.
(An earlier version spliced decoded thumbnails into the text stream as vision
blocks; that made a *preview* a second, competing reference and let one mod's
preview bleed into another's subject.  The latent alone is the honest
reference.)

Subject binding (optional ``auto_bind``)
────────────────────────────────────────
The model was trained on prompts of the form::

    subject_definitions:
    <Subject 1> is the excited blonde in the magenta bikini from <Picture 1>, ...
    <Subject 2> is the shy blonde in the teal bikini from <Picture 2>, ...

The binding glue is the ``from <Picture N>`` / ``from <Video N>`` clause — it
ties a subject to a specific ref block (positionally).  With ``auto_bind`` on,
this node prepends such a block built from the mods themselves (slot order =
ordinal, the mod's stored ``description`` is the label, ``<Audio j>`` is added
when the mod carries audio).  These are pure text tags — no vision blocks.

Usage
─────
  Load H3 RefMods ──mods──> MiniMax H3 RefMods to Video ──positive──> Sampler
                       (clip, prompt, width/height/length)

Keep your subject descriptions in the same order you loaded the mods (slot 1 =
<Subject 1> / <Video 1>, etc.).  The console prints the slot -> tag mapping.
"""

from __future__ import annotations

import torch

from comfy_api.latest import io

from ..py.core import H3RefMod


def _mod_blocks(loads) -> list:
    """Ref blocks (the DiT payload — the mod latents) for a loader bundle.

    Each block comes straight from ``mod.ref_block(strength)``: an image mod is
    a 1-frame video block, a mod with audio is a video_audio block (its audio
    rows pack immediately before its video rows).  The DiT attends to these
    blocks directly.
    """
    ref_blocks: list = []
    for mod, strength in loads:
        block = mod.ref_block(strength)
        if block is not None:
            ref_blocks.append(block)
    return ref_blocks


def _mod_tags(mod: H3RefMod, counters: dict) -> list:
    """Ordinal tag(s) for one mod, bumping the shared per-type counters.

    Mirrors the native tokenizer's independent counters and core.ref_block's
    kind mapping:  image -> <Picture i>;  video -> <Video k>;  a mod carrying
    audio also gets <Audio j> (audio label first, native order).
    """
    has_audio = mod.audio_latent is not None and mod.ref_audio_t > 0
    tags = []
    if has_audio:
        counters["audio"] += 1
        tags.append(f"<Audio {counters['audio']}>")
    # an image mod with audio is emitted as a 1-frame video block, so it tags as video
    if mod.kind == "video" or has_audio:
        counters["video"] += 1
        tags.append(f"<Video {counters['video']}>")
    else:
        counters["image"] += 1
        tags.append(f"<Picture {counters['image']}>")
    return tags


def _build_subject_preamble(loads) -> "tuple[str, list]":
    """Auto-generate MiniMax ``subject_definitions:`` bindings from the mods.

    Each mod carries everything needed: its slot position (the ordinal), its
    ``description`` (the human label), and whether it has audio (``ref_audio_t``).
    One ``<Subject N>`` per mod, in slot order, bound to its ref tag (and voice
    tag if present) via the trained ``from <Video N>`` clause.  Pure text — the
    latent stays the only reference.  Returns (preamble_text, per_mod_tags).
    """
    counters = {"image": 0, "audio": 0, "video": 0}
    lines = []
    per_mod_tags = []
    for i, (mod, _s) in enumerate(loads, start=1):
        tags = _mod_tags(mod, counters)
        per_mod_tags.append((mod.name, tags))
        label = (mod.description or "").strip() or mod.name
        video_tag = next((t for t in tags if t.startswith(("<Video", "<Picture"))), None)
        audio_tag = next((t for t in tags if t.startswith("<Audio")), None)
        line = f"<Subject {i}> is {label}"
        if video_tag:
            line += f" from {video_tag}"
        if audio_tag:
            line += f"; her voice is {audio_tag}"
        lines.append(line + ".")
    preamble = ""
    if lines:
        preamble = "subject_definitions:\n" + "\n".join(lines) + "\n\n"
    return preamble, per_mod_tags


class H3RefModsToVideo(io.ComfyNode):
    """ref2va with RefMods: prompt + a loader bundle -> conditioning + AV latent.

    The RefMod latent is the only reference — nothing is re-encoded and no
    preview pixels enter the text encoder.  With ``auto_bind`` on, a
    ``subject_definitions:`` block built from the mods is prepended to the
    prompt so each subject binds to its ref by ordinal tag.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RefModsToVideo",
            display_name="MiniMax H3 RefMods to Video",
            category="model/conditioning/minimax",
            description="Reference-to-Video using saved RefMods. The mod latent is the "
                        "ONLY reference — nothing re-encoded, no preview pixels in the "
                        "text encoder. With auto_bind on, a subject_definitions block "
                        "built from the mods is prepended so each subject binds to its "
                        "ref by ordinal tag (<Video 1>, <Audio 1>). Keep your subject "
                        "descriptions in the same order you loaded the mods.",
            inputs=[
                io.Clip.Input("clip"),
                io.Custom("H3_REF_MODS").Input("mods",
                    tooltip="RefMod bundle from Load H3 RefMods. Slot order = subject/tag "
                            "order; describe subjects in the same order in your prompt."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                    tooltip="Prompt. With auto_bind on, a subject_definitions block is "
                            "prepended automatically; describe the scene and reference "
                            "subjects as <Subject 1>, <Subject 2>, etc."),
                io.Int.Input("width", default=1344, min=32, max=8192, step=32),
                io.Int.Input("height", default=768, min=32, max=8192, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                    tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid "
                            "(124 = ~5s; trained range is ~124-362)."),
                io.Boolean.Input("auto_bind", default=True,
                    label_on="auto-bind subjects", label_off="no tags",
                    tooltip="Prepend a subject_definitions block built from the mods "
                            "themselves: each mod's ordinal tag (<Video 1>, <Audio 1>) "
                            "tied to its stored description via the trained 'from <Video N>' "
                            "clause. Pure text, no pixels. Turn off for tagless "
                            "content-based binding only."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, mods, prompt, width, height, length, auto_bind=True) -> io.NodeOutput:
        import node_helpers
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent

        latent, _frame_count = _empty_av_latent(width, height, length)

        # per-mod strength comes straight from the loader rows
        loads = [(m, s) for m, s in (mods or []) if m is not None and s > 0.0]
        if not loads:
            raise ValueError("H3RefModsToVideo: connect a Load H3 RefMods bundle "
                             "with at least one mod at strength > 0.")

        ref_blocks = _mod_blocks(loads)

        # optionally prepend the auto-generated subject bindings (pure text tags
        # tied to each mod's description), then tokenize.  No vision blocks.
        preamble, per_mod_tags = _build_subject_preamble(loads)
        full_prompt = (preamble + prompt) if auto_bind else prompt
        tokens = clip.tokenize(full_prompt)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})

        print("[H3RefModsToVideo] slot -> tag mapping:")
        for name, tags in per_mod_tags:
            print(f"  {name} -> {' + '.join(tags) if tags else '(no tag)'}")
        print(f"[H3RefModsToVideo] {len(ref_blocks)} ref block(s) injected"
              + (f"; auto-bind preamble:\n{preamble}" if auto_bind and preamble else
                 " (no tags — content-based binding)"))
        return io.NodeOutput(cond, latent)


NODE_CLASS_MAPPINGS = {
    "H3RefModsToVideo": H3RefModsToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModsToVideo": "MiniMax H3 RefMods to Video",
}
