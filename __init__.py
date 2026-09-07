"""
ComfyUI-MiniMaxH3Mod — no-training "RefMod" reference adapters for MiniMax H3

Reference videos are expensive because they inject thousands of tokens into
the H3 packed sequence.  A RefMod compresses a reference image/video into a
tiny pooled latent (~4-16 tokens) that still rides the model's native ref2va
path — the DiT attends to it through all 50 blocks like a real reference, but
at a fraction of the compute.  Extraction needs only the H3 VAE: no diffusion
model load, no training.

Nodes
─────
  H3RefModExtract       — ref_image_1..N stills + ref_video_1..N clips -> saved mod
  H3RefModFolderLoader  — load every image/video in a folder as an ordered ref list
  H3RefModsLoader       — load 1-8 mods with a typed strength each (LoRA-style)
  H3RefModsAxis         — A/B mod pairs on one signed slider each (negative -> A, positive -> B)
  H3RefModApply         — inject the bundle into MINIMAX_H3_COND (pack) or CONDITIONING (built-in);
                          the old split Apply/ApplyCond merged into one node (old workflows migrate)

"""

__original_author__ = "Luisa (luisacaotica)"
__maintainer__      = "FranckB"
__version__         = "0.1.2"

import os
import sys

from .nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .nodes import loader as _loader
from .nodes import refmods_to_video as _r2v
from .nodes import extract_from_folder as _eff
from .nodes import decode_refmod as _dec

NODE_CLASS_MAPPINGS = {
    **NODE_CLASS_MAPPINGS,
    **_loader.NODE_CLASS_MAPPINGS,
    **_r2v.NODE_CLASS_MAPPINGS,
    **_eff.NODE_CLASS_MAPPINGS,
    **_dec.NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **NODE_DISPLAY_NAME_MAPPINGS,
    **_loader.NODE_DISPLAY_NAME_MAPPINGS,
    **_r2v.NODE_DISPLAY_NAME_MAPPINGS,
    **_eff.NODE_DISPLAY_NAME_MAPPINGS,
    **_dec.NODE_DISPLAY_NAME_MAPPINGS,
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
