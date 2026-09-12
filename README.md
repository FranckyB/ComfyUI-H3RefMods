# ComfyUI-H3RefMods

Fork of [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod)

Compact RefMod tools for MiniMax H3, fork is tuned more deliberately toward identity workflows while keeping the compatibility with original add-on.  They can both be used at the same time.

This fork aims to make character work simpler, with faster picking, simpler apply flow and easier refMod creation from dataset folders. The more advanced option can be used by installing the original add-on.


<p align="center">
  <img src="docs/examples/browser_example.png" alt="Extract H3 RefMod in use" />
  <br />
  <em>Example using refMods from <a href="https://huggingface.co/spaces/malcolmrey/browser">Malcolm Reynolds</a>.</em>
</p>

<p align="center">
  <img src="docs/examples/workflow_example.png" alt="Workflow example" />
  <br />
  <em>Workflow example.</em>
</p>

<p align="center">
  <img src="docs/examples/create_from_folder_example.png" alt="Create from folder example" />
  <br />
  <em>Create refMods from Folder example.</em>
</p>

## Main additions

- `Visual RefMod Picker` lets you browse `models/refmods` visually with previews and append a selected RefMod bundle to an incoming stack. When matching `*_Video` and `*_Audio` files exist for the same RefMod, they are shown as one picker item and loaded together.
- `Visual RefMod Picker` also exposes separate video/audio weight controls for the two halves of a paired RefMod. A weight from `0` to `1` behaves like the old strength control, while values above `1` repeat the same RefMod as extra copies, so `2.7` means two full copies plus one `0.7` copy.
- `Create H3 RefMod From Folder` scans a folder of images, video, and optional audio, writes metadata, and saves upstream-compatible split RefMods as `*_Video.safetensors` and `*_Audio.safetensors` when audio is present. It can also batch-create one RefMod set per immediate subfolder and copy a matching thumbnail beside the saved mod.
- `Create H3 RefMod From Inputs` creates a RefMod directly from wired image, video, and optional audio inputs.
- `Load RefMod` and `Load RefMod Stack` load one or more saved RefMods for reuse and now use the same weight behavior: `0..1` is regular strength, values above `1` expand into repeated copies plus a fractional tail.
- `Apply H3 RefMod` is the streamlined apply node for the usual identity case, using a flat full-length reference curve.
- `Apply H3 RefMod Advanced` keeps the fuller timing and shaping controls for more deliberate modulation.
- `Apply H3 RefMod Axis` lets you feed in two `Visual RefMod Picker` selections and treat them like a signed slider for video and audio independently: negative values select A, positive values select B, and `0` means no effect for that channel.
- `H3 RefMod Step Curve` shapes reference strength across denoising steps rather than across the video timeline.
Added in - `tools` a `generate_video_thumbnails.py` script, that can generate same-name `.png` thumbnails from `.mp4` files in a folder by grabbing a random frame between 25% and 75% of each clip. This is just a small helper to speed up building large RefMod collections when your videos do not already have preview images.  The random part, is so you can generate again, if unhappy with the chosen frame. (Needs ffmpeg and ffprobe)

## Notes

- This fork keeps prior RefMod behavior and compatibility in place, but have been set to default to an identity-oriented workflow.
- Split `*_Video` and `*_Audio` RefMods are designed to stay compatible with `ComfyUI-MiniMaxH3Mod`, while the local picker/browser groups them as one logical item for convenience.

More advanced explaination can be found at [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod)

## Installation

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/FranckyB/ComfyUI-H3RefMods
```