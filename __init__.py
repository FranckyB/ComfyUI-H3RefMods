"""
ComfyUI-H3RefModPicker — a set of nodes for picking H3 reference mods in ComfyUI as well as batch creation.
"""

__author__      = "FranckB"
__version__         = "0.1.5"

from .nodes import refmod_apply
from .nodes import refmod_loader
from .nodes import refmod_visual_picker
from .nodes import refmod_create
from .nodes import refmod_axis

NODE_CLASS_MAPPINGS = {
    **refmod_apply.NODE_CLASS_MAPPINGS,
    **refmod_loader.NODE_CLASS_MAPPINGS,
    **refmod_visual_picker.NODE_CLASS_MAPPINGS,
    **refmod_create.NODE_CLASS_MAPPINGS,
    **refmod_axis.NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **refmod_apply.NODE_DISPLAY_NAME_MAPPINGS,
    **refmod_loader.NODE_DISPLAY_NAME_MAPPINGS,
    **refmod_visual_picker.NODE_DISPLAY_NAME_MAPPINGS,
    **refmod_create.NODE_DISPLAY_NAME_MAPPINGS,
    **refmod_axis.NODE_DISPLAY_NAME_MAPPINGS,
}
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
