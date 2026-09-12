import { app } from "../../scripts/app.js";

const TARGET_NODES = new Set([
    "H3RefModCreateFromFolder",
    "H3RefModCreateFromInputs",
]);

const MIN_NODE_WIDTH = 320;

const COMMON_PRESET_FIELDS = [
    "concept_type",
    "audio_concept_type",
    "mode",
    "ref_resolution",
    "pool_h",
    "pool_w",
    "latent_frames",
    "max_tokens",
    "identity",
    "merge",
    "motion_only",
    "multiplier",
    "audio_max_seconds",
    "audio_max_tokens",
    "audio_budget_policy",
    "max_total_tokens",
    "budget_policy",
    "max_frames",
];

const PRESET_CONFIG = {
    H3RefModCreateFromFolder: {
        manual: { values: {}, fields: [] },
        identity_encode: {
            values: {
                concept_type: "identity",
                audio_concept_type: "voice",
                mode: "Full Reference",
                ref_resolution: 1024,
                pool_h: 16,
                pool_w: 16,
                latent_frames: 16,
                max_tokens: 8192,
                identity: 0,
                merge: false,
                motion_only: false,
                multiplier: 1,
                max_frames: 240,
                audio_max_seconds: 30.0,
                audio_max_tokens: 5120,
                audio_budget_policy: "error",
                max_total_tokens: 0,
                budget_policy: "truncate",
            },
            fields: COMMON_PRESET_FIELDS,
        },
        style_experimental: {
            values: {
                concept_type: "style",
                audio_concept_type: "voice",
                mode: "Compressed Reference",
                ref_resolution: 1024,
                pool_h: 8,
                pool_w: 8,
                latent_frames: 16,
                max_tokens: 8192,
                identity: 150,
                merge: false,
                motion_only: false,
                multiplier: 1,
                max_frames: 240,
                audio_max_seconds: 30.0,
                audio_max_tokens: 5120,
                audio_budget_policy: "error",
                max_total_tokens: 0,
                budget_policy: "truncate",
            },
            fields: COMMON_PRESET_FIELDS,
        },
        motion_sequence: {
            values: {
                concept_type: "pose_motion",
                audio_concept_type: "voice",
                mode: "Compressed Reference",
                ref_resolution: 1024,
                pool_h: 16,
                pool_w: 16,
                latent_frames: 16,
                max_tokens: 8192,
                identity: 500,
                merge: false,
                motion_only: false,
                multiplier: 1,
                max_frames: 240,
                audio_max_seconds: 30.0,
                audio_max_tokens: 5120,
                audio_budget_policy: "error",
                max_total_tokens: 0,
                budget_policy: "truncate",
            },
            fields: COMMON_PRESET_FIELDS,
        },
    },
    H3RefModCreateFromInputs: {
        manual: { values: {}, fields: [] },
        identity_encode: {
            values: {
                concept_type: "identity",
                audio_concept_type: "voice",
                mode: "Full Reference",
                ref_resolution: 1024,
                pool_h: 16,
                pool_w: 16,
                latent_frames: 16,
                max_tokens: 5120,
                identity: 0,
                merge: false,
                motion_only: false,
                multiplier: 1,
                audio_max_seconds: 30.0,
                audio_max_tokens: 5120,
                audio_budget_policy: "error",
                max_total_tokens: 0,
                budget_policy: "truncate",
            },
            fields: COMMON_PRESET_FIELDS.filter((name) => name !== "max_frames"),
        },
        style_experimental: {
            values: {
                concept_type: "style",
                audio_concept_type: "voice",
                mode: "Compressed Reference",
                ref_resolution: 1024,
                pool_h: 8,
                pool_w: 8,
                latent_frames: 16,
                max_tokens: 5120,
                identity: 150,
                merge: false,
                motion_only: false,
                multiplier: 1,
                audio_max_seconds: 30.0,
                audio_max_tokens: 5120,
                audio_budget_policy: "error",
                max_total_tokens: 0,
                budget_policy: "truncate",
            },
            fields: COMMON_PRESET_FIELDS.filter((name) => name !== "max_frames"),
        },
        motion_sequence: {
            values: {
                concept_type: "pose_motion",
                audio_concept_type: "voice",
                mode: "Compressed Reference",
                ref_resolution: 1024,
                pool_h: 16,
                pool_w: 16,
                latent_frames: 16,
                max_tokens: 5120,
                identity: 500,
                merge: false,
                motion_only: false,
                multiplier: 1,
                audio_max_seconds: 30.0,
                audio_max_tokens: 5120,
                audio_budget_policy: "error",
                max_total_tokens: 0,
                budget_policy: "truncate",
            },
            fields: COMMON_PRESET_FIELDS.filter((name) => name !== "max_frames"),
        },
    },
};

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget?.name === name) || null;
}

function moveWidgetAfter(node, widgetName, afterName) {
    if (!node.widgets) return;
    const widget = findWidget(node, widgetName);
    const after = findWidget(node, afterName);
    if (!widget || !after) return;
    const from = node.widgets.indexOf(widget);
    const afterIndex = node.widgets.indexOf(after);
    if (from < 0 || afterIndex < 0) return;
    node.widgets.splice(from, 1);
    const insertAt = node.widgets.indexOf(after) + 1;
    node.widgets.splice(insertAt, 0, widget);
}

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (!widget.__h3VisibilityWrapped) {
        widget.__h3VisibilityWrapped = true;
        widget.__h3BaseComputeSize = widget.computeSize;
        widget.computeSize = function (...args) {
            if (widget.__h3Hidden) {
                return [0, -4];
            }
            if (typeof widget.__h3BaseComputeSize === "function") {
                return widget.__h3BaseComputeSize.apply(this, args);
            }
            return [MIN_NODE_WIDTH, Number(LiteGraph?.NODE_WIDGET_HEIGHT || 24)];
        };
    }
    if (typeof widget.__h3OriginalDisplay === "undefined" && widget.inputEl) {
        widget.__h3OriginalDisplay = widget.inputEl.style.display;
    }
    widget.__h3Hidden = !visible;
    widget.hidden = !visible;
    if (widget.inputEl) {
        widget.inputEl.style.display = visible ? (widget.__h3OriginalDisplay || "") : "none";
    }
}

function refreshNodeLayout(node) {
    if (Array.isArray(node.size)) {
        node.size[0] = Math.max(Number(node.size[0]) || 0, MIN_NODE_WIDTH);
    }
    const nextWidth = Math.max(node.size?.[0] || 0, MIN_NODE_WIDTH);
    const nextHeight = node.computeSize?.()?.[1] || node.size?.[1] || 0;
    node.setSize?.([nextWidth, nextHeight]);
    node.onResize?.(node.size);
    node.setDirtyCanvas(true, true);
    app.graph?.setDirtyCanvas(true, true);
}

function syncPresetValues(node) {
    const presetWidget = findWidget(node, "extraction_preset");
    const preset = String(presetWidget?.value || "manual");
    const nodeType = String(node.comfyClass || "");
    const nodePresets = PRESET_CONFIG[nodeType] || PRESET_CONFIG.H3RefModCreateFromInputs;
    const config = nodePresets[preset] || nodePresets.manual;
    for (const [name, value] of Object.entries(config.values)) {
        const widget = findWidget(node, name);
        if (!widget) continue;
        widget.value = value;
        if (widget.inputEl) {
            widget.inputEl.value = String(value);
            if (typeof widget.inputEl.checked === "boolean" && typeof value === "boolean") {
                widget.inputEl.checked = value;
            }
        }
    }
}

function applyFolderNamingVisibility(node) {
    if (String(node.comfyClass || "") !== "H3RefModCreateFromFolder") return;
    const useSubfoldersWidget = findWidget(node, "use_subfolders");
    const useFolderAsNameWidget = findWidget(node, "use_folder_as_name");
    const nameWidget = findWidget(node, "name");
    const useSubfolders = Boolean(useSubfoldersWidget?.value);
    const useFolderAsName = Boolean(useFolderAsNameWidget?.value);

    setWidgetVisible(useFolderAsNameWidget, !useSubfolders);
    setWidgetVisible(nameWidget, !useSubfolders && !useFolderAsName);
}

function applyPresetVisibility(node) {
    const presetWidget = findWidget(node, "extraction_preset");
    const preset = String(presetWidget?.value || "manual");
    const nodeType = String(node.comfyClass || "");
    const nodePresets = PRESET_CONFIG[nodeType] || PRESET_CONFIG.H3RefModCreateFromInputs;
    const active = new Set((nodePresets[preset] || nodePresets.manual).fields);
    const known = new Set(Object.values(nodePresets).flatMap((config) => config.fields));
    for (const name of known) {
        setWidgetVisible(findWidget(node, name), !(preset !== "manual" && active.has(name)));
    }
    applyFolderNamingVisibility(node);
    refreshNodeLayout(node);
}

function attachWidgetRefresh(node, widgetName, onAfterChange) {
    const widget = findWidget(node, widgetName);
    if (!widget || widget.__h3RefreshAttached) return;
    widget.__h3RefreshAttached = true;
    const originalCallback = widget.callback;
    widget.callback = function () {
        const result = originalCallback?.apply(this, arguments);
        onAfterChange();
        return result;
    };
}

function enhanceCreateNode(node) {
    if (node.__h3CreateUiEnhanced) return;
    node.__h3CreateUiEnhanced = true;

    const originalOnResize = node.onResize;
    node.onResize = function (size) {
        if (size) {
            size[0] = Math.max(size[0] || 0, MIN_NODE_WIDTH);
        }
        return originalOnResize?.apply(this, arguments);
    };

    moveWidgetAfter(node, "extraction_preset", "name");
    moveWidgetAfter(node, "budget_policy", "max_total_tokens");
    syncPresetValues(node);
    applyPresetVisibility(node);

    attachWidgetRefresh(node, "extraction_preset", () => {
        syncPresetValues(node);
        applyPresetVisibility(node);
    });
    attachWidgetRefresh(node, "use_subfolders", () => applyPresetVisibility(node));
    attachWidgetRefresh(node, "use_folder_as_name", () => applyPresetVisibility(node));

    const onConfigure = node.onConfigure;
    node.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        moveWidgetAfter(this, "extraction_preset", "name");
        moveWidgetAfter(this, "budget_policy", "max_total_tokens");
        syncPresetValues(this);
        applyPresetVisibility(this);
        attachWidgetRefresh(this, "extraction_preset", () => {
            syncPresetValues(this);
            applyPresetVisibility(this);
        });
        attachWidgetRefresh(this, "use_subfolders", () => applyPresetVisibility(this));
        attachWidgetRefresh(this, "use_folder_as_name", () => applyPresetVisibility(this));
        return result;
    };
}

app.registerExtension({
    name: "H3RefModPicker.RefModCreate",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGET_NODES.has(nodeData?.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            enhanceCreateNode(this);
            return result;
        };
    },
});

console.log("[H3RefModPicker] RefModCreate extension loaded");