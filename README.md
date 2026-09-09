# ComfyUI-H3RefMods

Fork of [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod)

Compact RefMod tools for MiniMax H3, fork is tuned more deliberately toward identity workflows while keeping the earlier stacking, blending, and curve-driven features intact.

The recent direction of this fork has been to make everyday character and persona work simpler: faster picking, simpler apply flows, and easier RefMod creation from dataset folders. At the same time, the more advanced controls are still here when you want to layer multiple mods, shape their influence over time, or drive them through the existing H3 conditioning path.

## What this pack now focuses on

- Identity-first workflows for saved RefMods.
- Simpler application with `Apply H3 RefMod` for the common case.
- Advanced control still available with `Apply H3 RefMod Advanced` and step-curve tools.
- Easier browsing with the new `Visual RefMod Picker`.
- Easier creation with `Create H3 RefMod From Folder`, including subfolder batch creation for whole datasets.

## Main nodes

- `Create H3 RefMod From Folder` scans a folder of images, video, and optional audio, builds a RefMod, writes metadata, and can batch-create one RefMod per immediate subfolder. It also copies a matching thumbnail beside the saved mod.
- `Extract H3 RefMod` creates a RefMod directly from wired image, video, and optional audio inputs.
- `Load H3 RefMods` loads one or more saved RefMods as a bundle for reuse.
- `Visual RefMod Picker` lets you browse `models/refmods` visually with previews and append a selected RefMod to an incoming bundle.
- `Apply H3 RefMod` is the streamlined apply node for the usual identity case, using a flat full-length reference curve.
- `Apply H3 RefMod Advanced` keeps the fuller timing and shaping controls for more deliberate modulation.
- `H3 RefMod Step Curve` shapes reference strength across denoising steps rather than across the video timeline.

## Notes

- This fork keeps prior RefMod behavior and compatibility in place where practical, but the node set has been geared up around a clearer identity-oriented workflow.
- Local H3 VAE loading is supported, so creation no longer depends on pulling VAE internals from an external MiniMax H3 custom-node pack.
- The visual browser is intentionally scoped to `models/refmods` and its subfolders.

Usage details, background, and longer explanations from original add-on in [docs/README.md](docs/README.md).
