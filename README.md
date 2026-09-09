# ComfyUI-H3RefMods

Fork of [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod)

Compact RefMod tools for MiniMax H3, fork is tuned more deliberately toward identity workflows while keeping the compatibility with original add-on.  They can both be used at the same time.

This fork aims make character work simpler, with faster picking, simpler apply flow and easier RefMod creation from dataset folders. The more advanced option can be used by installing the original add-on.

## Main additions

- `Visual RefMod Picker` lets you browse `models/refmods` visually with previews and append a selected RefMod bundle to an incoming stack. When matching `*_Video` and `*_Audio` files exist for the same RefMod, they are shown as one picker item and loaded together.
- `Visual RefMod Picker` also exposes separate strength controls for the visual and audio halves of a paired RefMod, so you can weaken video identity without also weakening the paired voice, or vice versa.
- `Create H3 RefMod From Folder` scans a folder of images, video, and optional audio, writes metadata, and saves upstream-compatible split RefMods as `*_Video.safetensors` and `*_Audio.safetensors` when audio is present. It can also batch-create one RefMod set per immediate subfolder and copy a matching thumbnail beside the saved mod.
- `Create H3 RefMod From Inputs` creates a RefMod directly from wired image, video, and optional audio inputs.
- `Load RefMod` and `Load RefMod Stack` loads one or more saved RefMods for reuse.
- `Apply H3 RefMod` is the streamlined apply node for the usual identity case, using a flat full-length reference curve.
- `Apply H3 RefMod Advanced` keeps the fuller timing and shaping controls for more deliberate modulation.
- `Apply H3 RefMod Axis` lets you feed in two `Visual RefMod Picker` selections and treat them like a signed slider for video and audio independently: negative values select A, positive values select B, and `0` means no effect for that channel.
- `H3 RefMod Step Curve` shapes reference strength across denoising steps rather than across the video timeline.

## Notes

- This fork keeps prior RefMod behavior and compatibility in place where practical, but the node set has been geared up around a clearer identity-oriented workflow.
- Local H3 VAE loading is supported, so creation no longer depends on pulling VAE internals from an external MiniMax H3 custom-node pack.
- Split `*_Video` and `*_Audio` RefMods are designed to stay compatible with `ComfyUI-MiniMaxH3Mod`, while the local picker/browser groups them as one logical item for convenience.
- The visual browser is intentionally scoped to `models/refmods` and its subfolders.

Usage details, background, and longer explanations from original add-on in [docs/README.md](docs/README.md).
