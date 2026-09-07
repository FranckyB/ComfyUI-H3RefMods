"""
extract_from_folder.py — H3RefModExtractFolder node.

Folder-driven RefMod extraction, the in-graph counterpart to the
``tools/generate_refmod.py`` + ``tools/extract_mod.py`` CLI pipeline.

Point it at a dataset folder, give the mod a name and concept type, and it
scans the folder for reference images / videos / audio, encodes them with the
connected H3 VAEs, embeds a Qwen "name tag" thumbnail strip, and saves the
``.safetensors`` mod — all inside ComfyUI, no subprocess, reusing the
already-loaded VAE.

This is the folder-driven sibling of ``H3RefModExtract`` (which takes refs as
IMAGE tensors).  Here the refs come from a folder scan instead of wired-in
tensors, and — unlike the tensor-driven Extract node — audio is embedded too,
so a persona mod carries its voice for the RefMods-to-Video subject tags.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch

import comfy.utils
from comfy_api.latest import io

from ..py.common import (
    list_media_files,
    load_image_file,
    load_video_file,
    refmods_dir,
)
from ..py.core import (
    CONCEPT_TYPES,
    H3RefMod,
    aspect_grid,
    fit_token_budget,
    normalize_mode,
    optimize_latent,
    pool_latent,
)
# reuse the encode helpers from the tensor-driven Extract node
from . import nodes as _nodes_mod  # the pack's own nodes.py (for _MOD_LIST_CACHE_KEY)
from .nodes import (
    _ensure_min_size,
    _MOD_CACHE,
    _MOD_CACHE_MAX,
    _resize_ref,
    _sanitize_name,
    _snap_to_causal_grid,
    _summarize,
)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg", ".opus"}


# ═══════════════════════════════════════════════════════════════════════════
# Folder scanning (images / videos / audio)
# ═══════════════════════════════════════════════════════════════════════════

def _scan_folder(folder: str) -> "tuple[List[str], List[str], List[str]]":
    """(images, videos, audios) directly under ``folder``, sorted by name."""
    images, videos = list_media_files(folder)
    audios = []
    if os.path.isdir(folder):
        for fn in sorted(os.listdir(folder)):
            if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                p = os.path.join(folder, fn)
                if os.path.isfile(p):
                    audios.append(p)
    return images, videos, audios


# ═══════════════════════════════════════════════════════════════════════════
# Audio loading + encoding (ported from tools/extract_mod.py)
# ═══════════════════════════════════════════════════════════════════════════

def _load_audio_waveform(path: str) -> "tuple[torch.Tensor, int]":
    """Load an audio file -> (waveform [C, L] float32, sample_rate)."""
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform.float(), sr
    except Exception:
        pass
    try:
        import soundfile as sf
        data, sr = sf.read(path, always_2d=True)  # [L, C]
        return torch.from_numpy(data.T).float(), sr
    except Exception:
        pass
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    t = torch.from_numpy(data).float()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    else:
        t = t.T
    if data.dtype.kind in ("i", "u"):
        t = t / float(2 ** (8 * data.dtype.itemsize - 1))
    return t, sr


def _resample(waveform: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    """Resample [C, L] -> [C, L'] via torchaudio, falling back to scipy."""
    try:
        import torchaudio
        return torchaudio.functional.resample(waveform, sr, target_sr)
    except Exception:
        import numpy as np
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, target_sr)
        out = resample_poly(waveform.numpy(), target_sr // g, sr // g, axis=-1)
        return torch.from_numpy(np.ascontiguousarray(out)).float()


def _encode_ref_audio(audio_vae, waveform: torch.Tensor, sr: int, device) -> torch.Tensor:
    """Encode a waveform to an H3 audio latent [1, 32, 2, T] (see CLI extract_mod)."""
    import comfy.model_management
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = _resample(waveform, sr, vae_sr)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)      # mono -> stereo
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    batch = waveform.unsqueeze(0)             # [1, 2, L]
    model = audio_vae.first_stage_model
    last_err = None
    for dev in (device, torch.device("cpu")):
        try:
            if dev.type == "cuda":
                comfy.model_management.load_models_gpu([audio_vae.patcher])
            else:
                model.to(dev)
            with torch.no_grad():
                z = model.encode(batch.to(dev)).float().cpu()
            if dev.type == "cuda":
                try:
                    audio_vae.patcher.unpatch_model()
                except Exception:
                    pass
            return z
        except (torch.OutOfMemoryError, RuntimeError) as e:
            last_err = e
            if "out of memory" not in str(e).lower() and "OOM" not in str(e):
                raise
            continue
    raise last_err


# ═══════════════════════════════════════════════════════════════════════════
# Node
# ═══════════════════════════════════════════════════════════════════════════

class H3RefModExtractFolder(io.ComfyNode):
    """Create a RefMod from every image/video/audio in a folder.

    The in-graph version of ``generate_refmod.py``: scans a dataset folder,
    encodes the media with the connected H3 VAEs, embeds audio as the mod's
    voice, and saves the ``.safetensors`` mod.  Defaults to an ``identity``
    concept in ``encode`` mode — the right choice for a person/character.

    This is a terminal (output-less) node: it saves the mod to disk and that's
    it.  Load the result afterward with Load H3 RefMods.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RefModCreate",
            display_name="Create H3 RefMod",
            category="H3RefMod",
            description="Scan a dataset folder for reference images/videos/audio and "
                        "create a RefMod (.safetensors). Defaults to an 'identity' "
                        "concept in 'encode' mode — the right choice for a person/"
                        "character. Audio files in the folder are embedded as the mod's "
                        "voice (needs the audio VAE). Saves to models/refmods/ by default.",
            inputs=[
                io.String.Input("folder", default="",
                    tooltip="Dataset folder with reference images/videos/audio. REQUIRED — "
                            "an absolute path, or a folder name inside ComfyUI's input/ "
                            "directory. The node will NOT run if left empty."),
                io.String.Input("name", default="my_concept",
                    tooltip="Saved mod name (appears in the Load H3 RefMods dropdown after a "
                            "reload)."),
                io.Combo.Input("concept_type", options=list(CONCEPT_TYPES), default="identity",
                    tooltip="What this mod represents. 'identity' (default) = a specific "
                            "person/character; also pose_motion, clothing, background, style, "
                            "generic."),
                io.Combo.Input("mode", options=["encode", "training"], default="encode",
                    tooltip="'encode' (default) = full-res VAE encode, max identity (~1K "
                            "tokens/img) — recommended for people. 'training' = pooled grid "
                            "refined by 'identity' steps — cheaper, concept/motion only."),
                io.Vae.Input("vae",
                    tooltip="The MiniMax H3 video VAE."),
                io.Vae.Input("audio_vae", optional=True,
                    tooltip="The MiniMax H3 audio VAE — needed to embed audio files found in "
                            "the folder. If not connected, audio is skipped with a note."),
                io.Int.Input("ref_resolution", default=1024, min=256, max=2048, step=64,
                    tooltip="Target short edge in px (downscale only). 1024 default; 2048 = "
                            "official max fidelity, 4x the tokens."),
                io.Int.Input("max_tokens", default=8192, min=0, max=65536, step=512,
                    tooltip="Hard cap on total injected tokens (0 = no cap). Near-duplicate "
                            "frames dropped first, then resampled to fit."),
                io.Int.Input("identity", default=500, min=0, max=2000, step=50,
                    tooltip="Training mode only: gradient refinement steps (0 = pure pooling)."),
                io.Int.Input("max_frames", default=240, min=2, max=4800, step=1,
                    tooltip="Video frames kept per video (uniformly sampled during decode)."),
                io.String.Input("description", default="", multiline=True,
                    tooltip="Optional text describing the concept, stored in the mod and "
                            "emitted by the loaders' prompt_hint output."),
                io.String.Input("save_dir", default="", optional=True,
                    tooltip="Where to save the mod. Empty = ComfyUI models/refmods/ (default, "
                            "recommended so the loaders find it). Set a path to save elsewhere."),
                io.Boolean.Input("save", default=True, label_on="save", label_off="don't save",
                    tooltip="Save the mod to disk so Load H3 RefMods can pick it up later."),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, folder, name, mode, concept_type, vae, audio_vae=None,
                ref_resolution=1024, max_tokens=8192, identity=500, max_frames=240,
                description="", save_dir="", save=True) -> io.NodeOutput:
        from .nodes import _resolve_folder
        if not (folder or "").strip().strip('"'):
            raise ValueError(
                "Create H3 RefMod: 'folder' is empty. Point it at a dataset folder "
                "(absolute path, or a name inside input/) — the node refuses to run "
                "without one so it can't accidentally scan your whole input/ directory.")
        folder = _resolve_folder(folder)
        name = _sanitize_name(name)
        mode = normalize_mode(mode)

        images, videos, audios = _scan_folder(folder)
        if not images and not videos:
            raise ValueError(
                f"H3RefModExtractFolder: no images or videos found in '{folder}'. "
                "Extraction needs at least one image or video reference.")
        print(f"[H3RefModExtractFolder] {folder}: {len(images)} image(s), "
              f"{len(videos)} video(s), {len(audios)} audio file(s)")

        device = comfy.model_management.get_torch_device()

        # ── audio identity (optional) ────────────────────────────────────
        audio_latent = None
        ref_audio_t = 0
        if audios:
            if audio_vae is not None:
                aframes = []
                for ap in audios:
                    waveform, sr = _load_audio_waveform(ap)
                    z = _encode_ref_audio(audio_vae, waveform, sr, device)
                    print(f"[H3RefModExtractFolder] audio {os.path.basename(ap)}: "
                          f"latent {tuple(z.shape)}")
                    aframes.append(z.to(torch.float16))
                audio_latent = torch.cat(aframes, dim=-1) if len(aframes) > 1 else aframes[0]
                ref_audio_t = audio_latent.shape[-1]
            else:
                print(f"[H3RefModExtractFolder] {len(audios)} audio file(s) found but no "
                      "audio_vae connected — audio NOT embedded.")

        # ── load visual refs as tensors ──────────────────────────────────
        sources = []  # (tensor [T,H,W,3], is_video)
        for p in images:
            sources.append((load_image_file(p, max_edge=ref_resolution * 2), False))
        for p in videos:
            sources.append((load_video_file(p, max_frames=max_frames,
                                            max_edge=ref_resolution * 2), True))

        # shared spatial canvas (encode mode) / pool grid (training mode)
        canvas = None
        if mode == "encode" and len(sources) > 1:
            h, w = sources[0][0].shape[1], sources[0][0].shape[2]
            scale = min(1.0, ref_resolution / min(h, w))
            canvas = (max(32, round(w * scale / 32) * 32),
                      max(32, round(h * scale / 32) * 32))
        pool_grid = None
        if mode == "training":
            h0, w0 = sources[0][0].shape[1], sources[0][0].shape[2]
            pool_grid = aspect_grid(16, 16, h0 / w0)
        gh, gw = pool_grid if pool_grid is not None else (16, 16)

        # ── encode each source ───────────────────────────────────────────
        frames = []
        source_shapes = []
        n_img = n_vid = 0
        n_refs = len(sources)
        pbar = comfy.utils.ProgressBar(n_refs)
        for idx, (src, is_video) in enumerate(sources):
            label = f"ref {idx + 1}/{n_refs} ({'video' if is_video else 'image'})"
            if not is_video:
                src = src[:1]  # pin stills to a single frame
            src = _resize_ref(src, ref_resolution, canvas)
            src = _ensure_min_size(src)
            if is_video and src.shape[0] > 1:
                valid_t = _snap_to_causal_grid(src.shape[0])
                if valid_t != src.shape[0]:
                    src = src[:valid_t]
            with torch.no_grad():
                z = vae.encode(src)
            if z.dim() != 5 or z.shape[1] != 24:
                raise ValueError(f"Expected a MiniMax H3 video VAE latent [1,24,T,H,W], "
                                 f"got {tuple(z.shape)}. The connected VAE is not the H3 VAE.")
            source_shapes.append(f"{z.shape[2]}x{z.shape[3]}x{z.shape[4]}")
            if mode == "encode":
                pooled = z.to(torch.float16)
            else:
                pool_t = min(16, z.shape[2]) if is_video else 1
                pooled = pool_latent(z, pool_t, gh, gw).to(torch.float16)
                if identity > 0:
                    pooled = optimize_latent(pooled, z.float(), steps=int(identity),
                                             progress_every=100)
            frames.append(pooled)
            n_vid += 1 if is_video else 0
            n_img += 0 if is_video else 1
            print(f"[H3RefModExtractFolder] {label}: encoded {tuple(pooled.shape)}")
            pbar.update_absolute(idx + 1)

        latent = torch.cat(frames, dim=2)
        if max_tokens > 0:
            latent = fit_token_budget(latent, max_tokens, name)
        total_t = latent.shape[2]
        kind = "video" if total_t > 1 else "image"
        px_w, px_h = latent.shape[4] * 16, latent.shape[3] * 16

        mod = H3RefMod(
            name=name,
            kind=kind,
            latent=latent,
            latent_h=latent.shape[3],
            latent_w=latent.shape[4],
            latent_t=total_t,
            mode=mode,
            source="stack" if len(frames) > 1 else ("video" if n_vid else "image"),
            source_shape=" +".join(source_shapes),
            pool=(f"full-res {px_w}x{px_h}px (short-edge cap {ref_resolution}px)"
                  if mode == "encode" else f"{total_t}x{gh}x{gw}"),
            optimize_steps=int(identity) if mode == "training" else 0,
            tags=[f"{n_img} img, {n_vid} vid"]
                 + ([f"{ref_audio_t} audio"] if ref_audio_t > 0 else []),
            description=(description or "").strip(),
            concept_type=concept_type,
            audio_latent=audio_latent,
            ref_audio_t=ref_audio_t,
        )

        out_dir = save_dir.strip().strip('"') or refmods_dir()
        if save:
            path = mod.save(os.path.join(out_dir, name))
            _MOD_CACHE[name] = mod
            if len(_MOD_CACHE) > _MOD_CACHE_MAX:
                _MOD_CACHE.pop(next(iter(_MOD_CACHE)))
            _nodes_mod._MOD_LIST_CACHE_KEY = None  # refresh the loader dropdown
            print(f"[CreateH3RefMod] saved {_summarize(mod)} -> {path}")
        else:
            print(f"[CreateH3RefMod] {_summarize(mod)} (not saved)")
        return io.NodeOutput()


NODE_CLASS_MAPPINGS = {
    "H3RefModCreate": H3RefModExtractFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModCreate": "Create H3 RefMod",
}
