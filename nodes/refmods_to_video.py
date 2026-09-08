"""
refmods_to_video.py — H3RefModsToVideo node.

A RefMod-aware counterpart to ComfyUI's native "MiniMax H3 Reference to Video"
(``comfy_extras.nodes_minimax_h3.MiniMaxH3ReferenceToVideo``), and a
convenience merge of the built-in ``CLIPTextEncode`` + ``Apply H3 RefMod``:
one node, prompt and mods connected in the same place.

Design — content-based binding, same mechanism as Apply H3 RefMod
────────────────────────────────────────────────────────────────────────────
Earlier versions of this node tried increasingly elaborate tokenizer-side
tag grounding: first bare ``<Subject N>``/``<Video k>`` text tags, then real
pixel thumbnails fed to ``clip.tokenize(minimax_ref_items=...)`` so the CLIP
vision encoder could actually see each reference, then the full MiniMax
six-section rewrite format (``subject_definitions`` / ``summary`` /
``retention_analysis`` / ``detailed_description``) to match the model's
documented training shape.  None of it fixed the actual symptom: audio
still ended up attached to the wrong subject's video.

That's the tell that tags were never the problem.  ``H3RefMod.ref_block()``
already packs a mod's audio and video into ONE combined ``video_audio``
block (audio rows immediately before video rows, same block dict) — this is
exactly what the old, tag-less ``H3RefModApply`` node has always injected,
and it has the same audio/video mismatch.  Since Apply never tokenized
anything special (it injects ``minimax_refs`` onto an already-encoded
conditioning) and still showed the same symptom, the mismatch isn't a text/
vision-grounding problem — it lives elsewhere (most likely the DiT's own
content-based attention struggling to route multiple similar reference
blocks, the same fidelity/compression story documented in
``H3RefModExtract``: a heavily compressed ``training``-mode mod carries far
less identity signal than a real reference).

So this node now does exactly what Apply does — plain
``clip.tokenize(prompt)``, then append each mod's ``ref_block(strength)`` to
``minimax_refs`` — just merged into one node so you don't need a separate
``CLIPTextEncode`` + ``Apply H3 RefMod`` chain.  No tags, no thumbnails, no
subject_definitions preamble; the RefMod latent (and, for a mod with audio,
its bundled voice) is the only reference, exactly like Apply.

Usage
─────
  Load H3 RefMods ──mods──> MiniMax H3 RefMods to Video ──positive──> Sampler
                       (clip, prompt, width/height/length)
"""

from __future__ import annotations

from comfy_api.latest import io

from .nodes import _normalize_mods_input, _unpack_row

def _mod_blocks(loads) -> list:
    """Ref blocks (the DiT payload — the mod latents) for a loader bundle.

    Each block comes straight from ``mod.ref_block(strength)``: an image mod is
    a 1-frame video block, a mod with audio is a video_audio block (its audio
    rows pack immediately before its video rows).  The DiT attends to these
    blocks directly — same mechanism as Apply H3 RefMod.
    """
    ref_blocks: list = []
    for mod, strength, _desc in loads:
        block = mod.ref_block(strength)
        if block is not None:
            ref_blocks.append(block)
    return ref_blocks


class H3RefModsToVideo(io.ComfyNode):
    """ref2va with RefMods: prompt + a loader bundle -> conditioning + AV latent.

    Plain content-based binding — the same mechanism ``Apply H3 RefMod``
    uses (each mod's ``ref_block(strength)`` appended to ``minimax_refs``;
    a mod with audio carries it bundled in the same block, right next to its
    video).  No tags, no vision-encoder grounding, no rewrite-format
    preamble — this node is just ``CLIPTextEncode`` + ``Apply H3 RefMod``
    merged into one, so the prompt and the mods are connected in the same
    place.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RefModsToVideo",
            display_name="MiniMax H3 RefMods to Video",
            category="model/conditioning/minimax",
            description="Reference-to-Video using saved RefMods — a merge of the built-in "
                        "CLIPTextEncode + Apply H3 RefMod into one node. Tokenizes your "
                        "prompt as-is, then appends each mod's ref_block (its DiT-side "
                        "latent, with any audio bundled into the same block) to "
                        "minimax_refs. Same content-based binding as Apply H3 RefMod, just "
                        "with the prompt and mods connected in the same place.",
            inputs=[
                io.Clip.Input("clip"),
                io.MultiType.Input("mods",
                    types=[io.Custom("H3_REFMOD"), io.Custom("H3_REF_MODS")],
                    tooltip="A single RefMod (from Load H3 RefMod) or a bundle (from Load H3 "
                            "RefMods / Load H3 RefMod Axis / Combine H3 RefMods / Extract H3 "
                            "RefMod) — connect either directly, no Combine node needed for "
                            "just one mod."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                    tooltip="Prompt, tokenized as-is (no tags or preamble added)."),
                io.Int.Input("width", default=1344, min=32, max=8192, step=32),
                io.Int.Input("height", default=768, min=32, max=8192, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                    tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid "
                            "(124 = ~5s; trained range is ~124-362)."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, mods, prompt, width, height, length) -> io.NodeOutput:
        import node_helpers
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent

        latent, _frame_count = _empty_av_latent(width, height, length)

        # per-mod strength comes straight from the loader rows; normalize to
        # (mod, strength, description_override) once — description isn't used
        # here (no tags/preamble), kept only so both loader row shapes
        # (2-tuple / 3-tuple) work interchangeably.
        loads = [_unpack_row(row) for row in _normalize_mods_input(mods)]
        loads = [row for row in loads if row[0] is not None and row[1] > 0.0]
        if not loads:
            raise ValueError("H3RefModsToVideo: connect a Load H3 RefMod (single) or a "
                             "Load H3 RefMods bundle with at least one mod at strength > 0.")

        ref_blocks = _mod_blocks(loads)

        tokens = clip.tokenize(prompt)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})

        print(f"[H3RefModsToVideo] " + ", ".join(f"{mod.name}@{strength:.2f}" for mod, strength, _d in loads)
              + f" ({len(ref_blocks)} ref block(s) injected, content-based binding)")
        return io.NodeOutput(cond, latent)


NODE_CLASS_MAPPINGS = {
    "H3RefModsToVideo": H3RefModsToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModsToVideo": "MiniMax H3 RefMods to Video",
}

