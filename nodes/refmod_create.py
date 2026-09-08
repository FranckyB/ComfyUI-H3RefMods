"""
create_refmod.py — H3RefModCreateFromFolder + H3RefModExtract nodes.

Single location for both RefMod-creation methods:

  H3RefModCreateFromFolder  — folder-driven extraction:
                              Point it at a dataset folder,
                              give the mod a name and concept type, and it
                              scans the folder for reference media (images, videos, audio),
                              encodes them with the connected H3 VAEs, and saves the mod.

  H3RefModCreateFromInputs  — connection-driven extraction:
                              Connect media (images, videos, audio) directly.
                              Give the mod a name and concept type, and it
                              creates a saved mod from the connected inputs.


Both scan/encode their refs with the connected H3 VAE(s) and save a
``.safetensors`` mod, output as an ``H3_REF_MODS`` bundle; the only
difference is where the reference media comes from (a folder scan vs.
wired-in tensors).  Both are flagged as output nodes (``is_output_node=True``)
so they actually run standalone, without anything connected downstream.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch

import comfy.utils
from comfy_api.latest import io

from ..py.refmod_common import (
    list_media_files,
    load_image_file,
    load_video_file,
    refmods_dir,
)
from ..py.refmod_core import (
    CONCEPT_TYPES,
    H3RefMod,
    aspect_grid,
    fit_token_budget,
    normalize_mode,
    optimize_latent,
    pool_latent,
)
# reuse the encode helpers shared with the loader/mods-listing side of the pack
from . import refmod_apply as _nodes_mod  # the pack's own nodes.py (for _MOD_LIST_CACHE_KEY)
from .refmod_apply import (
    _ensure_min_size,
    _h3_pack_submodule,
    _mask_latent,
    _MOD_CACHE,
    _MOD_CACHE_MAX,
    _normalize_mask_batch,
    _normalize_ref,
    _resize_mask,
    _resize_ref,
    _sanitize_name,
    _snap_to_causal_grid,
    _summarize,
    _THUMB_MAX_FRAMES,
    _THUMB_SHORT_EDGE,
    _unique_mod_path,
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


def _create_mod_from_folder(
    folder: str,
    name: str,
    mode: str,
    concept_type: str,
    vae,
    audio_vae=None,
    ref_resolution: int = 1024,
    max_tokens: int = 8192,
    identity: int = 500,
    max_frames: int = 240,
    description: str = "",
    save_dir: str = "",
    save: bool = True,
) -> H3RefMod:
    """Create (and optionally save) a single RefMod from one folder."""
    name = _sanitize_name(name)
    mode = normalize_mode(mode)

    images, videos, audios = _scan_folder(folder)
    if not images and not videos:
        raise ValueError(
            f"H3RefModCreateFromFolder: no images or videos found in '{folder}'. "
            "Extraction needs at least one image or video reference.")
    print(f"[H3RefModCreateFromFolder] {folder}: {len(images)} image(s), "
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
                print(f"[H3RefModCreateFromFolder] audio {os.path.basename(ap)}: "
                      f"latent {tuple(z.shape)}")
                aframes.append(z.to(torch.float16))
            audio_latent = torch.cat(aframes, dim=-1) if len(aframes) > 1 else aframes[0]
            ref_audio_t = audio_latent.shape[-1]
        else:
            print(f"[H3RefModCreateFromFolder] {len(audios)} audio file(s) found but no "
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
        print(f"[H3RefModCreateFromFolder] {label}: encoded {tuple(pooled.shape)}")
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
        saved_name, path_no_ext = _unique_mod_path(out_dir, name)
        if saved_name != name:
            print(f"[H3RefModCreateFromFolder] '{name}' already exists in {out_dir} "
                  f"— saving as '{saved_name}' instead (existing mods are never "
                  f"overwritten).")
            mod.name = saved_name
        path = mod.save(path_no_ext)
        _MOD_CACHE[saved_name] = mod
        if len(_MOD_CACHE) > _MOD_CACHE_MAX:
            _MOD_CACHE.pop(next(iter(_MOD_CACHE)))
        _nodes_mod._MOD_LIST_CACHE_KEY = None  # refresh the loader dropdown
        print(f"[CreateH3RefMod] saved {_summarize(mod)} -> {path}")
    else:
        print(f"[CreateH3RefMod] {_summarize(mod)} (not saved)")
    return mod


# ═══════════════════════════════════════════════════════════════════════════
# Node
# ═══════════════════════════════════════════════════════════════════════════

class H3RefModCreateFromFolder(io.ComfyNode):
    """Create a RefMod from every image/video/audio in a folder.

    The in-graph version of ``generate_refmod.py``: scans a dataset folder,
    encodes the media with the connected H3 VAEs, embeds audio as the mod's
    voice, and saves the ``.safetensors`` mod.  Defaults to an ``identity``
    concept in ``encode`` mode — the right choice for a person/character.

    Flagged as an output node (``is_output_node=True``) so it actually runs
    even when nothing is wired to its ``mods`` output — it still saves to
    disk either way.  ``mods`` carries the freshly created mod at strength
    1.0 (same ``H3_REF_MODS`` shape ``H3RefModExtract`` outputs), so you can
    chain it straight into Apply H3 RefMod / MiniMax H3 RefMods to Video /
    Combine H3 RefMods without a separate reload step — or just ignore it
    and reload later with Load H3 RefMods.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RefModCreateFromFolder",
            display_name="Create H3 RefMod From Folder",
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
                io.Boolean.Input("use_subfolders", default=False,
                    label_on="subfolders", label_off="single folder",
                    tooltip="When ON, the folder input is treated as a parent directory: "
                            "every immediate subfolder is scanned and turned into its own "
                            "RefMod, named after the subfolder. The 'name' input is ignored "
                            "in this mode. Great for batch-processing a dataset of concepts."),
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
                io.String.Input("save_dir", default="models/refmods/", optional=True,
                    tooltip="Where to save the mod. Empty = ComfyUI models/refmods/ (default, "
                            "recommended so the loaders find it). Set a path to save elsewhere. "
                            "If a mod with this name already exists there, it is never "
                            "overwritten — the save name gets '_2', '_3', etc. appended instead "
                            "(the console prints the final name used)."),
                io.Boolean.Input("save", default=True, label_on="save", label_off="don't save",
                    tooltip="Save the mod to disk so Load H3 RefMods can pick it up later."),
            ],
            outputs=[
                io.Custom("H3_REF_MODS").Output("mods",
                    tooltip="Bundle with the freshly created mod(s) at strength 1.0 — feed "
                            "straight into Apply H3 RefMod / MiniMax H3 RefMods to Video / "
                            "Combine H3 RefMods, no reload needed. In subfolder mode this "
                            "contains one mod per subfolder."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, folder, name, mode, concept_type, vae, audio_vae=None,
                ref_resolution=1024, max_tokens=8192, identity=500, max_frames=240,
                description="", save_dir="", save=True, use_subfolders=False) -> io.NodeOutput:
        from .refmod_apply import _resolve_folder
        if not (folder or "").strip().strip('"'):
            raise ValueError(
                "Create H3 RefMod: 'folder' is empty. Point it at a dataset folder "
                "(absolute path, or a name inside input/) — the node refuses to run "
                "without one so it can't accidentally scan your whole input/ directory.")
        folder = _resolve_folder(folder)
        mode = normalize_mode(mode)

        if use_subfolders:
            subfolders = [
                os.path.join(folder, d)
                for d in sorted(os.listdir(folder))
                if os.path.isdir(os.path.join(folder, d))
            ]
            if not subfolders:
                raise ValueError(
                    f"H3RefModCreateFromFolder: use_subfolders=True but no subfolders "
                    f"found in '{folder}'.")
            print(f"[H3RefModCreateFromFolder] subfolder mode: {len(subfolders)} folder(s)")
            mods = []
            for sub in subfolders:
                sub_name = _sanitize_name(os.path.basename(sub))
                mod = _create_mod_from_folder(
                    sub, sub_name, mode, concept_type, vae, audio_vae,
                    ref_resolution, max_tokens, identity, max_frames,
                    description, save_dir, save,
                )
                mods.append((mod, 1.0))
            return io.NodeOutput(mods)

        name = _sanitize_name(name)
        mod = _create_mod_from_folder(
            folder, name, mode, concept_type, vae, audio_vae,
            ref_resolution, max_tokens, identity, max_frames,
            description, save_dir, save,
        )
        return io.NodeOutput([(mod, 1.0)])


# ═══════════════════════════════════════════════════════════════════════════
# Node: H3RefModExtract (V3 — Autogrow reference inputs)
# ═══════════════════════════════════════════════════════════════════════════

class H3RefModCreateFromInputs(io.ComfyNode):
    """
    Turn one or more references of the same concept into a RefMod.

    Refs are added with the "+" button: stills plug into ``ref_image_1``,
    video frames into ``ref_video_1``, and the next slot of that type appears.
    Each ref is one row of the same concept (different angle / expression /
    setting / a dance move); they are stacked into a video-kind mod.

    Two modes:

      * ``encode`` (default) — each ref is resized to ``ref_resolution`` short
        edge (down only) and VAE-encoded at that resolution, exactly like the
        official ref2video node.  The mod stores the real encode, so identity
        (a face, an outfit) comes through; files are ~0.2-1 MB per frame.
        (Old name: ``full``.)
      * ``training`` — each ref is first resized to ``ref_resolution`` short
        edge too (the latent is pooled to a tiny grid anyway, so encoding at
        native resolution is wasted compute — this is the main speed dial for
        training mode), then average-pooled to a tiny grid (4x4 = 4 tokens
        per frame) and refined with gradient steps against the encode — still
        no diffusion model.  Nearly free to inject but only carries concept /
        motion, not fine identity.  (Old name: ``pooled``.)

    ``identity`` (training mode only) is the refinement loop — the only
    "training" in the pack.

    ``max_tokens`` (0 = off) hard-caps the total injected tokens: when the
    stacked refs exceed it, near-duplicate latent frames are dropped first,
    then frames are resampled to fit (see ``core.fit_token_budget``).

    Flagged as an output node (``is_output_node=True``) so it runs standalone
    even when nothing is wired to its ``mods`` output — it still saves to
    disk either way.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RefModCreateFromInputs",
            display_name="Create H3 RefMod From Inputs",
            description=(
                "Turn one or more references of the same concept into a RefMod. "
                "Stills plug into ref_image_1, video frames into ref_video_1, "
                "and the next slot of that type appears. refs are stacked into "
                "one video-kind mod, so a multi-image moodboard keeps each ref's "
                "own content instead of averaging away. 'training' mode (default) "
                "compresses the refs to a grid and refines it — good identity at "
                "a fraction of the tokens; 'encode' stores the full-res encode "
                "(max identity, MB-size mod, ~1K tokens/img)."
            ),
            category="H3RefMod",
            inputs=[
                io.String.Input("name", default="my_concept",
                    tooltip="Saved mod name (appears in the Load H3 RefMods dropdown after a reload)."),
                io.Combo.Input("mode", options=["training", "encode"],
                    default="training",
                    tooltip="'training' (default) = compressed grid refined by the 'identity' "
                            "dial — a good balance of identity vs tokens. 'encode' = straight "
                            "full-res VAE encode (max identity, MB-size mod, ~1K tokens/img). "
                            "Old mods saved as 'full'/'pooled' still load and normalize to "
                            "these two."),
                io.Combo.Input("concept_type", options=list(CONCEPT_TYPES), default="generic",
                    tooltip="What this mod represents — 'identity' (a specific person/character), "
                            "'pose_motion' (a pose/dance/gesture/camera move), 'clothing', "
                            "'background', 'style', or 'generic'. Stored in the mod and used by "
                            "the loaders' prompt_hint output (merges concept_type + description "
                            "into a string you can concat onto your CLIP prompt). 'identity' in "
                            "training mode with a small grid also triggers a warning nudging you "
                            "toward 'encode' mode or a bigger grid — pooling is lossy in exactly "
                            "the way that destroys facial identity."),
                io.Autogrow.Input("refs_image", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip="Reference still: one image of the "
                            "concept (angle / expression / outfit). Optional — leave empty when using "
                            "video refs and/or a folder bundle."),
                        prefix="ref_image_", min=0, max=16)),
                io.Autogrow.Input("refs_video", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", tooltip="Reference video frames "
                            "[T,H,W,C] (a multi-frame batch = a video ref with motion). "
                            "Optional — leave empty when using image refs and/or a folder bundle."),
                        prefix="ref_video_", min=0, max=8)),
                io.Custom("H3_REF_LIST").Input("refs_bundle", optional=True,
                    tooltip="All images/videos from a Load H3 RefMod Folder node, appended after "
                            "the autogrow refs (bulk extraction)."),
                io.Mask.Input("mask", optional=True,
                    tooltip="Subject mask (or a batch, one per reference in order: images then "
                            "videos) marking what to keep at full weight. Everything outside the "
                            "mask collapses toward a heavily blurred copy of itself per spatial "
                            "cell (stays in-distribution — a flat noise-mix here decodes as a "
                            "woven/static texture instead of 'nothing'), controlled by "
                            "background_retention. Fixes 'encode' mode pulling in a background/style "
                            "that doesn't belong to the subject. A single mask broadcasts to every "
                            "reference; a batch must match the reference count."),
                io.Float.Input("background_retention", default=0.0, min=0.0, max=1.0, step=0.05,
                    tooltip="Only used when 'mask' is connected. Floor weight for the region "
                            "outside the mask: 0 = that region collapses to a heavily blurred "
                            "copy of itself (kills specific structure like a skyline/treeline "
                            "while staying smooth and in-distribution), 1 = mask has no effect. "
                            "Middle values (0.3-0.6) partially blur instead of fully."),
                io.Custom("MINIMAX_H3_AV_ENCODER").Input("av_encoder", optional=True,
                    tooltip="MiniMax-H3 VAE pack output (preferred; share the pack's VAE cache)."),
                io.Vae.Input("vae", optional=True,
                    tooltip="Standard VAE, used when av_encoder is not connected."),
                io.Int.Input("ref_resolution", default=1024, min=256, max=2048, step=64,
                    tooltip="Target short edge in px (downscale only, never upscale), applied to "
                            "BOTH modes: 'encode' stores at that res, 'training' encodes smaller "
                            "too (it pools to a grid anyway, so native-res encoding is wasted "
                            "compute — this is the main speed dial for training mode). 1024 is a "
                            "good default; 512 halves encode cost; 2048 = official max fidelity, "
                            "4x the tokens of 1024."),
                io.Int.Input("pool_h", default=16, min=2, max=64, step=2,
                    tooltip="Pooled mode: spatial latent grid after pooling. The grid is auto-fit to "
                            "the source's aspect ratio (long edge = max of the two dials, other edge "
                            "derived), so a portrait person isn't squished into a square grid "
                            "(the 'fat/chubby' distortion). Square sources keep the exact dial value. "
                            "16x16 = 64 tokens/frame (concept sweet spot); 32x32 = 256; 64x64 = 1024, "
                            "full-mode parity for identity."),
                io.Int.Input("pool_w", default=16, min=2, max=64, step=2,
                    tooltip="Pooled mode: grid width (long edge if the source is wider than tall)."),
                io.Int.Input("latent_frames", default=16, min=1, max=16,
                    tooltip="Frames kept per video ref: training mode pools them, encode mode uniformly "
                            "samples them (16x16x16 = 4096 tokens per video ref). Images always use 1."),
                io.Int.Input("identity", default=500, min=0, max=2000, step=50,
                    tooltip="Pooled mode only: how tightly the mod clings to the reference "
                            "(gradient refinement steps). Higher = more identity detail but sticks "
                            "to the refs' framing/background; lower = deviates from the refs but "
                            "loses detail. 500 is a good default; 0 = pure pooling."),
                io.Int.Input("multiplier", default=1, min=1, max=10, step=1,
                    tooltip="Data multiplier: repeat the extracted ref N times along time so a short "
                            "video/GIF (few tokens) isn't drowned out by the main video's tokens. "
                            "Each repeat duplicates the same latent frames, so attention weight on "
                            "the ref scales roughly with N. 1 = no repeat; file size grows with N."),
                io.Int.Input("max_tokens", default=5120, min=0, max=65536, step=512,
                    tooltip="Hard cap on the total tokens the mod injects (0 = no cap; 5120 is a good "
                            "performance default). If the stacked refs exceed it, near-duplicate "
                            "latent frames are dropped first (video refs are full of frames that "
                            "differ only by noise — each one still costs a token per spatial patch "
                            "in every block), then frames are resampled to fit. The cap is honored "
                            "after the multiplier. Lower latent_frames/ref_resolution instead to "
                            "avoid wasting encode work: ~23K tokens = one 1024px encode-mode video "
                            "ref at 16 frames."),
                io.String.Input("description", default="", multiline=True,
                    tooltip="Optional text describing the concept (e.g. 'a ginger woman with messy "
                            "hair', 'an animation style', 'handheld camera movement'). Stored in "
                            "the mod and printed in the info block — documentation only, no wiring."),
                io.Boolean.Input("save", default=True, label_on="save", label_off="don't save",
                    tooltip="Save the mod to ComfyUI models/refmods/ so Load H3 RefMods can pick "
                            "it up later. If a mod with this name already exists there, it is "
                            "never overwritten — the save name gets '_2', '_3', etc. appended "
                            "instead (the console prints the final name used)."),
            ],
            outputs=[
                io.Custom("H3_REF_MODS").Output("mods",
                    tooltip="Bundle with this one mod at strength 1.0. Feed it to Apply H3 RefMod "
                            "(or Load H3 RefMods after saving)."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, name, mode, refs_image=None, refs_video=None, refs_bundle=None,
                av_encoder=None, vae=None,
                ref_resolution=1024, pool_h=16, pool_w=16, latent_frames=16,
                identity=500, multiplier=1, max_tokens=0, description="", save=True,
                concept_type="generic", mask=None, background_retention=0.0,
                **legacy) -> io.NodeOutput:
        name = _sanitize_name(name)
        mode = normalize_mode(mode)  # accept legacy 'full'/'pooled'
        if concept_type == "identity" and mode == "training" and max(pool_h, pool_w) < 16:
            print(
                f"[H3RefModExtract] warning: concept_type='identity' with "
                f"mode='training' at a {pool_h}x{pool_w} grid — pooling averages away "
                f"exactly the detail that carries a face (this is almost certainly "
                f"your 'chubby/older' drift). For a person, either switch mode='encode' "
                f"(real identity, higher token cost) or raise pool_h/pool_w toward "
                f"32x32+ and expect it to still be a soft approximation, not a lock."
            )
        # old pre-Autogrow workflows pass their widget values through as kwargs:
        # map them onto the new inputs so those saved workflows keep running.
        # ``pool`` is the old height; ``pool_w`` arrives as the named param.
        if legacy.get("optimize") is not None:
            identity = legacy["optimize"]
        if legacy.get("pool") is not None:
            pool_h = int(legacy["pool"])
            if pool_w == 16:  # old single-pool default: square grid
                pool_w = pool_h
        if av_encoder is None and vae is None:
            raise ValueError(
                "H3RefModExtract: connect an av_encoder (MiniMax-H3 "
                "VAE loader) or a standard VAE.")
        pack = None
        if av_encoder is not None:
            vae_pack_mod = _h3_pack_submodule("models.vae")
            pack = vae_pack_mod.load_vae_pack(av_encoder.video_path, av_encoder.audio_path)

        # each Autogrow arrives as a dict keyed by its slot names
        # (ref_image_1..N / ref_video_1..N); videos stay multi-frame, images are
        # pinned to a single still.  Legacy workflows used flat image /
        # ref_image_N / ref_video_N inputs; a folder bundle is appended last.
        ordered = []
        for key in sorted((refs_image or {}).keys(),
                          key=lambda k: int(k.rsplit("_", 1)[1])):
            src = (refs_image or {})[key]
            if src is not None:
                ordered.append((src, False))
        for key in sorted((refs_video or {}).keys(),
                          key=lambda k: int(k.rsplit("_", 1)[1])):
            src = (refs_video or {})[key]
            if src is not None:
                ordered.append((src, True))
        legacy_keys = sorted(
            (k for k in legacy if k == "image" or k.startswith("ref_image_")
             or k.startswith("ref_video_")),
            key=lambda k: (0 if k == "image" else 1 if k.startswith("ref_image_") else 2,
                           int(k.rsplit("_", 1)[1]) if "_" in k else 0))
        for key in legacy_keys:
            if legacy[key] is not None:
                ordered.append((legacy[key], key.startswith("ref_video_")))
        if refs_bundle is not None:
            for src in refs_bundle:
                if src is not None:
                    norm = _normalize_ref(src, label="folder reference")
                    ordered.append((norm, norm.shape[0] > 1))
        if not ordered:
            raise ValueError(
                "H3RefModExtract: connect at least one image to "
                "ref_image_1, or video frames to ref_video_1, or a folder bundle.")
        sources = []
        for i, (src, is_video) in enumerate(ordered):
            norm = _normalize_ref(src, label=f"reference {i + 1}")
            if is_video:
                sources.append((norm, norm.shape[0] > 1))
            else:
                # image slot: pin to a single still even if a batch arrived
                sources.append((norm[:1], False))
        # encode each source independently (full-res or pooled), then stack.
        # Full-res refs must share one spatial canvas so the stacked latent has
        # a single H/W: anchor on the first source, cover-crop the rest to it.
        canvas = None
        if mode == "encode" and len(sources) > 1:
            h, w = sources[0][0].shape[1], sources[0][0].shape[2]
            scale = min(1.0, ref_resolution / min(h, w))
            canvas = (max(32, round(w * scale / 32) * 32),
                      max(32, round(h * scale / 32) * 32))
        # training mode: anchor the pool grid to the first source's aspect so
        # a portrait person isn't squished into a square 16x16 grid (the
        # "fat" distortion).  The VAE scales space uniformly, so pixel
        # aspect == latent aspect.
        pool_grid = None
        if mode == "training":
            h0, w0 = sources[0][0].shape[1], sources[0][0].shape[2]
            pool_grid = aspect_grid(pool_h, pool_w, h0 / w0)
            if pool_grid != (pool_h, pool_w):
                print(f"[H3RefModExtract] pooled grid {pool_h}x{pool_w} -> "
                      f"{pool_grid[0]}x{pool_grid[1]} to match source aspect "
                      f"{w0}x{h0} (avoids squishing the subject wide)")
        gh, gw = pool_grid if pool_grid is not None else (pool_h, pool_w)

        mask_batch = _normalize_mask_batch(mask, label="mask")
        if mask_batch is not None:
            if mask_batch.shape[0] == 1 and len(sources) > 1:
                mask_batch = mask_batch.expand(len(sources), -1, -1)
            elif mask_batch.shape[0] != len(sources):
                raise ValueError(
                    f"H3RefModExtract: mask has {mask_batch.shape[0]} entries but "
                    f"there are {len(sources)} references (images then videos, in order). "
                    f"Connect one mask (broadcasts to every ref) or exactly one per ref.")

        frames = []
        n_img = n_vid = 0
        source_shapes = []
        thumb_frames_list = []
        thumb_canvas = None
        n_refs = len(sources)
        pbar = comfy.utils.ProgressBar(n_refs)
        thumb_budget = max(1, _THUMB_MAX_FRAMES // max(1, n_refs))
        for src_idx in range(len(sources)):
            src, is_video = sources[src_idx]
            label = f"ref {src_idx + 1}/{n_refs} ({'video' if is_video else 'image'})"
            print(f"[H3RefModExtract] {label}: "
                  f"source {tuple(src.shape)}, mode={mode}"
                  + (f", identity={identity} steps" if mode == "training" and identity > 0 else ""))
            if mode == "encode":
                # downscale (never upscale) to the target short edge, sample
                # videos to latent_frames frames, then encode at full res
                if is_video and latent_frames < src.shape[0]:
                    idx = torch.linspace(0, src.shape[0] - 1, latent_frames).round().long()
                    src = src[idx]
                src = _resize_ref(src, ref_resolution, canvas)
            else:
                # training mode: encode smaller too — the latent is pooled
                # to a tiny grid anyway, so encoding at native resolution is
                # wasted compute. Resize preserves aspect, so the pool grid
                # anchored on the first source's aspect still applies.
                orig = (src.shape[1], src.shape[2])
                src = _resize_ref(src, ref_resolution, None)
                if (src.shape[1], src.shape[2]) != orig:
                    print(f"[H3RefModExtract] {label}: resized "
                          f"{orig[0]}x{orig[1]} -> {src.shape[1]}x{src.shape[2]} "
                          f"(ref_resolution={ref_resolution}) before encode")
            src = _ensure_min_size(src)
            # capture a small real-pixel thumbnail before it's consumed below,
            # for clip.tokenize(minimax_ref_items=...) grounding at Apply time
            # (see H3RefMod.thumb) — every source contributes at least one
            # frame so a multi-ref mod's thumbnail shows every angle, not just
            # the first.
            if thumb_canvas is None:
                th0, tw0 = src.shape[1], src.shape[2]
                scale0 = min(1.0, _THUMB_SHORT_EDGE / min(th0, tw0))
                thumb_canvas = (max(32, round(tw0 * scale0 / 32) * 32),
                                max(32, round(th0 * scale0 / 32) * 32))
            thumb_src = _resize_ref(src, _THUMB_SHORT_EDGE, thumb_canvas)
            if is_video and thumb_src.shape[0] > 1:
                n_pick = max(1, min(thumb_budget, thumb_src.shape[0]))
                idx = torch.linspace(0, thumb_src.shape[0] - 1, n_pick).round().long()
                thumb_src = thumb_src[idx]
            else:
                thumb_src = thumb_src[:1]
            thumb_frames_list.append(thumb_src.to(torch.float16))
            if is_video and src.shape[0] > 1:
                valid_t = _snap_to_causal_grid(src.shape[0])
                if valid_t != src.shape[0]:
                    print(f"[H3RefModExtract] reference {src_idx + 1} "
                          f"(video): trimming {src.shape[0]} -> {valid_t} frames "
                          f"to match the VAE's causal 4k+1 grid.")
                    src = src[:valid_t]
            mask_px = None
            if mask_batch is not None:
                mask_px = _resize_mask(mask_batch[src_idx:src_idx + 1], src.shape[1], src.shape[2])
            if src.shape[1] <= 0 or src.shape[2] <= 0:
                raise ValueError(
                    f"H3RefModExtract: reference {src_idx + 1} "
                    f"({'video' if is_video else 'image'}) has an empty frame "
                    f"{tuple(src.shape)} right before VAE encode (mode={mode}, "
                    f"ref_resolution={ref_resolution}, canvas={canvas}). "
                    f"Check that this specific reference's source image/video "
                    f"is valid.")
            # encode-path conventions differ:
            #  - av_encoder -> pack's raw H3 VAE: channel-first [1, 3, T, H, W]
            #    in [-1, 1] (same as the pack's own conditioning node)
            #  - vae -> comfy sd.VAE wrapper: channel-last [T, H, W, C] in [0, 1];
            #    the wrapper does its own layout conversion and /16 cropping, and
            #    would misread channel-first input (narrowing the channel dim to 0)
            if pack is not None:
                moved = src.movedim(-1, 1)
                if moved.shape[0] == 1:
                    pixels = moved
                else:
                    pixels = moved.permute(1, 0, 2, 3).unsqueeze(0)
                pixels = (pixels * 2.0 - 1.0).to(torch.float16)
                z = pack.encode_video(pixels)
            else:
                z = vae.encode(src)
            if z.dim() != 5 or z.shape[1] != 24:
                raise ValueError(
                    f"Expected a MiniMax H3 video VAE latent [1,24,T,H,W], "
                    f"got {tuple(z.shape)}. The connected VAE is not the H3 VAE.")
            source_shapes.append(f"{z.shape[2]}x{z.shape[3]}x{z.shape[4]}")

            if mask_px is not None:
                z = _mask_latent(z, mask_px, background_retention, seed_key=f"{name}:{src_idx}")
                print(f"[H3RefModExtract] {label}: applied subject mask "
                      f"(background_retention={background_retention})")

            if mode == "encode":
                pooled = z.to(torch.float16)
            else:
                pool_t = min(latent_frames, z.shape[2]) if is_video else 1
                gh, gw = pool_grid if pool_grid is not None else (pool_h, pool_w)
                pooled = pool_latent(z, pool_t, gh, gw).to(torch.float16)
                if identity > 0:
                    print(f"[H3RefModExtract] {label}: refining identity "
                          f"({int(identity)} gradient steps)...")
                    pooled = optimize_latent(pooled, z.float(), steps=int(identity),
                                              progress_every=100)
                    print(f"[H3RefModExtract] {label}: identity refinement done")
            frames.append(pooled)
            if is_video:
                n_vid += 1
            else:
                n_img += 1
            pbar.update_absolute(src_idx + 1)
            print(f"[H3RefModExtract] {label}: encoded "
                  f"{tuple(pooled.shape)} ({pooled.numel() * pooled.element_size() / 1024 / 1024:.2f} MB)")
            # drop the decoded source and the full-res latent as soon as we're
            # done with them, so a large folder doesn't keep every source +
            # every full encode resident while the remaining refs are encoded
            sources[src_idx] = None
            src = None
            z = None

        if mode == "encode" and identity > 0:
            print(f"[H3RefModExtract] warning: 'identity' only applies to "
                  f"training mode — encode mode stores the actual encode, so "
                  f"identity={identity} was ignored.")

        latent = torch.cat(frames, dim=2)  # [1, 24, total_t, h, w]
        if multiplier > 1:
            latent = latent.repeat(1, 1, multiplier, 1, 1)  # data multiplier
        if max_tokens > 0:
            latent = fit_token_budget(latent, max_tokens, name)
        total_t = latent.shape[2]
        kind = "video" if total_t > 1 else "image"
        # the VAE encodes at 16x spatial scale, so a latent of 40x20 = 640x320 px
        px_w, px_h = latent.shape[4] * 16, latent.shape[3] * 16
        thumb = torch.cat(thumb_frames_list, dim=0)
        if thumb.shape[0] > _THUMB_MAX_FRAMES:
            idx = torch.linspace(0, thumb.shape[0] - 1, _THUMB_MAX_FRAMES).round().long()
            thumb = thumb[idx]
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
            pool=f"full-res {px_w}x{px_h}px (short-edge cap {ref_resolution}px)" if mode == "encode" else f"{total_t}x{gh}x{gw}",
            optimize_steps=int(identity) if mode == "training" else 0,
            tags=[f"{n_img} img, {n_vid} vid"] + ([f"x{multiplier} repeat"] if multiplier > 1 else [])
                + ([f"masked (bg_retention={background_retention})"] if mask_batch is not None else []),
            description=(description or "").strip(),
            concept_type=concept_type,
            thumb=thumb,
        )

        if save:
            saved_name, path_no_ext = _unique_mod_path(refmods_dir(), name)
            if saved_name != name:
                print(f"[H3RefModCreateFromInputs] '{name}' already exists — saving as "
                      f"'{saved_name}' instead (existing mods are never overwritten).")
                mod.name = saved_name
            path = mod.save(path_no_ext)
            _MOD_CACHE[saved_name] = mod
            if len(_MOD_CACHE) > _MOD_CACHE_MAX:
                _MOD_CACHE.pop(next(iter(_MOD_CACHE)))
            _nodes_mod._MOD_LIST_CACHE_KEY = None  # new mod -> refresh the dropdown listing
            print(f"[H3RefModCreateFromInputs] saved {_summarize(mod)} -> {path}")
        else:
            print(f"[H3RefModCreateFromInputs] {_summarize(mod)} (not saved)")
        if mod.description:
            print(f"[H3RefModCreateFromInputs] description: {mod.description}")
        return io.NodeOutput([(mod, 1.0)])


NODE_CLASS_MAPPINGS = {
    "H3RefModCreateFromFolder": H3RefModCreateFromFolder,
    "H3RefModCreateFromInputs": H3RefModCreateFromInputs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModCreateFromFolder": "Create H3 RefMod From Folder",
    "H3RefModCreateFromInputs": "Create H3 RefMod From Inputs",
}
