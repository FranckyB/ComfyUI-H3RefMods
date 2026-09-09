from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple


_H3_VAE_CACHE: Dict[Tuple[str, str, str, bool], Tuple[Any, Optional[Any]]] = {}


def _cache_key(video_path: str, audio_path: str, device, load_audio: bool) -> Tuple[str, str, str, bool]:
    return (
        os.path.abspath(video_path or ""),
        os.path.abspath(audio_path or ""),
        str(device),
        bool(load_audio),
    )


def load_h3_vae_pair(video_path: str, audio_path: str = "", device=None,
                     load_audio: bool = True) -> Tuple[Any, Optional[Any]]:
    """Load and cache the H3 video/audio VAE wrappers from safetensors paths."""
    import comfy.model_management
    import comfy.sd
    import comfy.utils

    if not video_path:
        raise ValueError("load_h3_vae_pair: video_path is required")
    device = device or comfy.model_management.get_torch_device()
    key = _cache_key(video_path, audio_path, device, load_audio)
    cached = _H3_VAE_CACHE.get(key)
    if cached is not None:
        return cached

    sd, metadata = comfy.utils.load_torch_file(video_path, return_metadata=True)
    video_vae = comfy.sd.VAE(sd=sd, metadata=metadata, device=device)
    video_vae.throw_exception_if_invalid()

    audio_vae = None
    if load_audio and audio_path:
        asd, ameta = comfy.utils.load_torch_file(audio_path, return_metadata=True)
        audio_vae = comfy.sd.VAE(sd=asd, metadata=ameta, device=device)
        audio_vae.throw_exception_if_invalid()

    pair = (video_vae, audio_vae)
    _H3_VAE_CACHE[key] = pair
    return pair


def load_h3_vaes_from_av_encoder(av_encoder, device=None,
                                 load_audio: bool = True) -> Tuple[Any, Optional[Any]]:
    """Resolve a MiniMax H3 VAE-loader payload into local video/audio VAE wrappers."""
    video_path = getattr(av_encoder, "video_path", None)
    audio_path = getattr(av_encoder, "audio_path", None) or ""
    if not video_path:
        raise ValueError(
            "H3RefModExtract: av_encoder is missing video_path. Connect a valid "
            "MiniMax H3 VAE loader output."
        )
    return load_h3_vae_pair(video_path, audio_path, device=device,
                            load_audio=load_audio)