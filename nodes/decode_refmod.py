"""
decode_refmod.py — H3RefModDecode node.

Decode a saved RefMod's latent back into images, so you can see what the mod
actually captured.  Purely a debugging / inspection tool — it plays no part in
generation.

A RefMod stores a video-VAE latent [1, 24, T, h, w] (the reference).  This node
runs it back through the H3 video VAE decoder and returns the frames as an IMAGE
batch [T, H, W, 3], so you can preview a mod, sanity-check an extraction, or
eyeball what a pooled (training-mode) mod actually kept vs an encode-mode one.

Note: a training-mode (pooled) mod decodes to a tiny, blurry grid — that's the
point of the mode, it only stores a concept thumbnail.  An encode-mode mod
decodes to recognizable full-res frames.  Audio (if the mod has it) is not
decoded here — this is the visual reference only.
"""

from __future__ import annotations

import torch

import comfy.utils
from comfy_api.latest import io

from .nodes import _list_mod_names, _load_mod


class H3RefModDecode(io.ComfyNode):
    """Decode a RefMod's latent back to images for inspection."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RefModDecode",
            display_name="Decode H3 RefMod",
            category="H3RefMod",
            description="Decode a saved RefMod's latent back into images so you can see "
                        "what it captured. Inspection only — not part of generation. "
                        "An encode-mode mod decodes to recognizable frames; a "
                        "training-mode (pooled) mod decodes to a tiny blurry grid "
                        "(that's all it stores). Audio is not decoded.",
            inputs=[
                io.Combo.Input("mod", options=_list_mod_names(),
                    tooltip="The RefMod to decode (from models/refmods/)."),
                io.Vae.Input("vae",
                    tooltip="The MiniMax H3 video VAE."),
                io.Int.Input("max_frames", default=16, min=1, max=64, step=1,
                    tooltip="Cap on decoded frames (uniformly sampled if the mod has "
                            "more). Keeps a long mod from decoding hundreds of frames."),
            ],
            outputs=[
                io.Image.Output("images", display_name="images",
                    tooltip="The decoded reference frames [T, H, W, 3]."),
            ],
        )

    @classmethod
    def execute(cls, mod, vae, max_frames=16) -> io.NodeOutput:
        m = _load_mod(mod)
        z = m.latent  # [1, 24, T, h, w]
        total_t = z.shape[2]

        # uniform-sample down to max_frames frames before decode
        if total_t > max_frames:
            idx = torch.linspace(0, total_t - 1, max_frames).round().long()
            z = z[:, :, idx]

        with torch.no_grad():
            frames = vae.decode(z)
        # video latent decodes channel-last; frame count rides the batch dim, so a
        # 5-dim result is reshaped like the core VAEDecode node does
        if frames.dim() == 5:  # [B, T, H, W, C] -> [B*T, H, W, C]
            frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
        frames = frames.clamp(0.0, 1.0).float().cpu()

        print(f"[H3RefModDecode] '{m.name}' ({m.mode}, {m.kind}, "
              f"latent_t={total_t}) -> {tuple(frames.shape)}")
        return io.NodeOutput(frames)


NODE_CLASS_MAPPINGS = {
    "H3RefModDecode": H3RefModDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModDecode": "Decode H3 RefMod",
}
