import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PLACEHOLDER_IMAGE_PATH = new URL("./placeholder.png", import.meta.url).href;
const DEFAULT_PICKER_NODE_SIZE = [240, 480];

function isTypingTarget(target) {
    if (!target) return false;
    const tag = String(target.tagName || "").toUpperCase();
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || !!target.isContentEditable;
}

let _vrpHoveredNode = null;
let _vrpArrowListenerInstalled = false;

function isVrpNodeInGraph(node) {
    if (!node || !app.graph?._nodes) return false;
    return app.graph._nodes.includes(node);
}

function getActiveVisualRefModPickerNode() {
    if (_vrpHoveredNode && !_vrpHoveredNode.flags?.collapsed && isVrpNodeInGraph(_vrpHoveredNode)) {
        return _vrpHoveredNode;
    }
    if (_vrpHoveredNode && !isVrpNodeInGraph(_vrpHoveredNode)) {
        _vrpHoveredNode = null;
    }
    return null;
}

function installVrpArrowNavigation() {
    if (_vrpArrowListenerInstalled) return;
    _vrpArrowListenerInstalled = true;

    window.addEventListener("keydown", async (event) => {
        if (event.defaultPrevented) return;
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
        if (event.ctrlKey || event.altKey || event.metaKey || event.shiftKey) return;
        if (isTypingTarget(event.target)) return;

        const node = getActiveVisualRefModPickerNode();
        if (!node || typeof node._vrpStepModByDelta !== "function") return;

        const direction = event.key === "ArrowRight" ? 1 : -1;
        const handled = await node._vrpStepModByDelta(direction);
        if (!handled) return;
        event.preventDefault();
        event.stopPropagation();
    }, true);
}

function hideWidget(widget) {
    if (!widget) return;
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    if (widget.inputEl) widget.inputEl.style.display = "none";
}

function basenameForDisplay(value) {
    const normalized = String(value || "").replace(/\\/g, "/");
    const i = normalized.lastIndexOf("/");
    return i >= 0 ? normalized.substring(i + 1) : normalized;
}

function entryDisplayName(entry) {
    return String(entry?.name || basenameForDisplay(entry?.path || "") || "");
}

function dirnameForPath(value) {
    const normalized = String(value || "").replace(/\\/g, "/").replace(/\/+$/, "");
    const i = normalized.lastIndexOf("/");
    return i > 0 ? normalized.substring(0, i) : "";
}

function buildPreviewUrl(path) {
    if (!path) return PLACEHOLDER_IMAGE_PATH;
    return api.apiURL(`/h3refmods/refmod-browser/file?path=${encodeURIComponent(path)}`);
}

function installPreviewFallback(img) {
    if (!img || img.dataset.vrpFallbackInstalled === "1") return img;
    img.dataset.vrpFallbackInstalled = "1";
    img.addEventListener("error", () => {
        if (img.src === PLACEHOLDER_IMAGE_PATH) return;
        img.src = PLACEHOLDER_IMAGE_PATH;
    });
    return img;
}

function normalizePathKey(value) {
    return String(value || "").replace(/\\/g, "/").replace(/\/+/g, "/").toLowerCase();
}

function pathsEqual(a, b) {
    return normalizePathKey(a) === normalizePathKey(b);
}

async function getRefModsRoot() {
    try {
        const resp = await api.fetchApi("/h3refmods/refmod-browser/root");
        if (!resp.ok) return "";
        const data = await resp.json();
        return data.ok ? data.root : "";
    } catch (err) {
        console.warn("[VisualRefModPicker] Could not get refmods root:", err);
        return "";
    }
}

async function listBrowserFolder(path) {
    const query = path ? `?path=${encodeURIComponent(path)}` : "";
    const resp = await api.fetchApi(`/h3refmods/refmod-browser/list${query}`);
    if (!resp.ok) {
        let msg = `Request failed (${resp.status})`;
        try {
            const err = await resp.json();
            if (err?.error) msg = err.error;
        } catch {
            // ignore
        }
        throw new Error(msg);
    }
    return await resp.json();
}

function makeButton(label, onClick) {
    const button = document.createElement("button");
    button.textContent = label;
    button.style.cssText = `
        background: #2e3b4a;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 6px;
        color: #dce6f2;
        padding: 5px 12px;
        cursor: pointer;
        font-size: 12px;
    `;
    button.onmouseenter = () => {
        button.style.background = "#3a4a5c";
    };
    button.onmouseleave = () => {
        button.style.background = "#2e3b4a";
    };
    button.addEventListener("click", onClick);
    return button;
}

function createRefModBrowserModal(initialPath, onSelect) {
    const overlay = document.createElement("div");
    overlay.style.cssText = `
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
    `;

    const modal = document.createElement("div");
    modal.style.cssText = `
        background: rgba(40, 44, 52, 0.98);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 6px;
        width: 95%;
        max-width: 1304px;
        height: 95%;
        max-height: 1185px;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.45);
    `;

    const header = document.createElement("div");
    header.style.cssText = `
        padding: 15px 20px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    `;

    const topRow = document.createElement("div");
    topRow.style.cssText = "display: flex; justify-content: space-between; align-items: center;";

    const title = document.createElement("div");
    title.textContent = "Select RefMod";
    title.style.cssText = "margin: 0; color: #aaa; font-size: 1.17em; font-weight: 600;";

    const closeButton = document.createElement("button");
    closeButton.textContent = "×";
    closeButton.style.cssText = `
        background: none;
        border: none;
        color: #aaa;
        font-size: 24px;
        cursor: pointer;
        padding: 0;
        width: 30px;
        height: 30px;
    `;
    closeButton.onclick = () => overlay.remove();

    const upButton = makeButton("Up", () => {
        if (state.parentPath) void loadFolder(state.parentPath);
    });

    topRow.append(title, closeButton);
    header.append(topRow);

    const filterBar = document.createElement("div");
    filterBar.style.cssText = `
        padding: 10px 20px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        display: flex;
        align-items: center;
        gap: 10px;
    `;

    filterBar.appendChild(upButton);

    const searchWrap = document.createElement("div");
    searchWrap.style.cssText = "position: relative; flex: 1; display: flex; align-items: center;";

    const searchInput = document.createElement("input");
    searchInput.type = "text";
    searchInput.placeholder = "Search files...";
    searchInput.style.cssText = `
        flex: 1;
        padding: 8px 36px 8px 12px;
        background: rgba(45, 55, 72, 0.7);
        border: 1px solid rgba(226, 232, 240, 0.2);
        border-radius: 6px;
        color: #ccc;
    `;

    const clearSearchButton = document.createElement("button");
    clearSearchButton.type = "button";
    clearSearchButton.textContent = "×";
    clearSearchButton.title = "Clear search";
    clearSearchButton.style.cssText = `
        position: absolute;
        right: 8px;
        top: 50%;
        transform: translateY(-50%);
        width: 20px;
        height: 20px;
        padding: 0;
        border: none;
        background: transparent;
        color: #aaa;
        cursor: pointer;
        font-size: 16px;
        line-height: 1;
        display: none;
    `;

    searchWrap.append(searchInput, clearSearchButton);
    filterBar.appendChild(searchWrap);

    const body = document.createElement("div");
    body.style.cssText = `
        flex: 1;
        overflow-y: auto;
        padding: 20px;
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 15px;
        align-content: start;
    `;

    const status = document.createElement("div");
    status.style.cssText = "padding: 10px 20px; border-bottom: 1px solid rgba(255,255,255,0.1); display: flex; gap: 12px; align-items: center; font-size: 12px; color: #888;";

    const countsLabel = document.createElement("div");
    countsLabel.style.cssText = "white-space: nowrap;";

    const pathLabel = document.createElement("div");
    pathLabel.style.cssText = "min-width: 0; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;";

    status.append(countsLabel, pathLabel);

    modal.append(header, filterBar, status, body);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    overlay.addEventListener("click", (event) => {
        if (event.target === overlay) overlay.remove();
    });

    const state = {
        root: "",
        currentPath: "",
        parentPath: null,
        items: [],
    };

    function setCardSelected(card, selected) {
        if (!card) return;
        card.dataset.selected = selected ? "1" : "0";
        card.style.borderColor = selected ? "rgba(255, 255, 255, 0.95)" : "rgba(226, 232, 240, 0.2)";
        card.style.background = selected ? "rgba(50, 112, 163, 0.28)" : "rgba(45, 55, 72, 0.7)";
        card.style.transform = "translateY(0)";
    }

    function updateVisibleItems() {
        const search = String(searchInput.value || "").trim().toLowerCase();
        for (const entry of state.items) {
            const haystack = `${entry.kind === "dir" ? entry.data.name : entry.data.name}`.toLowerCase();
            const visible = !search || haystack.includes(search);
            entry.element.style.display = visible ? "flex" : "none";
        }
    }

    function updateSearchClearButton() {
        clearSearchButton.style.display = searchInput.value ? "block" : "none";
    }

    function folderCard(entry) {
        const card = document.createElement("div");
        card.className = "folder-item";
        card.style.cssText = `
            background: rgba(45, 55, 72, 0.7);
            border: 1px solid rgba(226, 232, 240, 0.2);
            border-radius: 6px;
            padding: 8px;
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            flex-direction: column;
            gap: 8px;
        `;
        const icon = document.createElement("div");
        icon.style.cssText = `
            width: 100%;
            height: 250px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 64px;
        `;
        icon.textContent = "📁";
        const label = document.createElement("div");
        label.textContent = entry.name;
        label.style.cssText = `
            font-size: 12px;
            color: #ccc;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            text-align: center;
        `;
        label.title = entry.name;
        card.append(icon, label);
        card.onmouseenter = () => {
            card.style.borderColor = "rgba(66, 153, 225, 0.9)";
            card.style.transform = "translateY(-2px)";
            card.style.background = "rgba(50, 112, 163, 0.5)";
        };
        card.onmouseleave = () => {
            card.style.borderColor = "rgba(226, 232, 240, 0.2)";
            card.style.transform = "translateY(0)";
            card.style.background = "rgba(45, 55, 72, 0.7)";
        };
        card.addEventListener("click", () => void loadFolder(entry.path));
        return card;
    }

    function modCard(entry) {
        const card = document.createElement("div");
        card.className = "thumbnail-item";
        card.style.cssText = `
            background: rgba(45, 55, 72, 0.7);
            border: 1px solid rgba(226, 232, 240, 0.2);
            border-radius: 6px;
            padding: 8px;
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            flex-direction: column;
            gap: 8px;
        `;
        const imageBox = document.createElement("div");
        imageBox.style.cssText = `
            width: 100%;
            height: 250px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            position: relative;
        `;
        const img = document.createElement("img");
        installPreviewFallback(img);
        img.src = entry.preview_path ? `${buildPreviewUrl(entry.preview_path)}&${Date.now()}` : PLACEHOLDER_IMAGE_PATH;
        img.style.cssText = `
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: center center;
            display: block;
            background: rgba(0, 0, 0, 0.5);
        `;
        imageBox.appendChild(img);
        const title = document.createElement("div");
        title.textContent = entryDisplayName(entry);
        title.style.cssText = `
            font-size: 12px;
            color: #ccc;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            text-align: center;
        `;
        title.title = entryDisplayName(entry);
        card.append(imageBox, title);
        const selectedPath = state.selectedPath;
        setCardSelected(card, !!selectedPath && pathsEqual(selectedPath, entry.path));
        card.onmouseenter = () => {
            if (card.dataset.selected === "1") return;
            card.style.borderColor = "rgba(66, 153, 225, 0.9)";
            card.style.transform = "translateY(-2px)";
            card.style.background = "rgba(50, 112, 163, 0.5)";
        };
        card.onmouseleave = () => {
            if (card.dataset.selected === "1") {
                setCardSelected(card, true);
                return;
            }
            card.style.borderColor = "rgba(226, 232, 240, 0.2)";
            card.style.transform = "translateY(0)";
            card.style.background = "rgba(45, 55, 72, 0.7)";
        };
        card.addEventListener("click", () => {
            onSelect(entry);
            overlay.remove();
        });
        return card;
    }

    async function loadFolder(path) {
        try {
            countsLabel.textContent = "Loading...";
            pathLabel.textContent = "";
            body.replaceChildren();
            state.items = [];
            const data = await listBrowserFolder(path);
            state.root = data.root || state.root;
            state.currentPath = data.current_path || path || state.currentPath;
            state.parentPath = data.parent_path || null;
            state.selectedPath = null;
            pathLabel.textContent = state.currentPath || state.root || "models/refmods";
            upButton.disabled = !state.parentPath;
            countsLabel.textContent = `${(data.dirs || []).length} folder(s), ${(data.mods || []).length} RefMod(s)`;

            for (const dir of data.dirs || []) {
                const element = folderCard(dir);
                state.items.push({ kind: "dir", data: dir, element });
                body.appendChild(element);
            }
            for (const mod of data.mods || []) {
                const element = modCard(mod);
                state.items.push({ kind: "mod", data: mod, element });
                body.appendChild(element);
            }
            if (!body.childNodes.length) {
                const empty = document.createElement("div");
                empty.textContent = "No RefMods or subfolders found here.";
                empty.style.cssText = "text-align: center; padding: 40px; color: #aaa;";
                body.appendChild(empty);
            }
            updateVisibleItems();
        } catch (err) {
            countsLabel.textContent = err?.message || "Could not load RefMods.";
            pathLabel.textContent = "";
        }
    }

    searchInput.addEventListener("input", () => {
        updateSearchClearButton();
        updateVisibleItems();
    });

    clearSearchButton.addEventListener("click", () => {
        searchInput.value = "";
        updateSearchClearButton();
        updateVisibleItems();
        searchInput.focus();
    });

    updateSearchClearButton();

    void (async () => {
        const root = await getRefModsRoot();
        state.root = root;
        await loadFolder(initialPath || root);
    })();
}

app.registerExtension({
    name: "H3RefModPicker.VisualRefModPicker",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "H3RefModVisualPicker") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;
            if (!node.properties) node.properties = {};
            node._configuredFromWorkflow = false;

            const modPathWidget = node.widgets?.find((w) => w.name === "mod_path");
            const strengthWidget = node.widgets?.find((w) => w.name === "video_weight")
                || node.widgets?.find((w) => w.name === "strength");
            const audioStrengthWidget = node.widgets?.find((w) => w.name === "audio_weight")
                || node.widgets?.find((w) => w.name === "audio_strength");

            const panel = document.createElement("div");
            panel.style.cssText = `
                display: flex;
                flex-direction: column;
                width: 100%;
                height: 100%;
                box-sizing: border-box;
                gap: 6px;
                overflow: hidden;
                background: transparent;
            `;

            const previewBox = document.createElement("div");
            previewBox.style.cssText = `
                position: relative;
                flex: 1;
                min-height: 80px;
                width: 100%;
                border-radius: 10px;
                background: rgba(34, 39, 48, 0.98);
                border: 1px solid rgba(78, 90, 108, 0.72);
                overflow: hidden;
                box-sizing: border-box;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 4px;
            `;

            const portraitFrame = document.createElement("div");
            portraitFrame.style.cssText = `
                position: relative;
                height: 100%;
                width: auto;
                max-width: 100%;
                max-height: 100%;
                aspect-ratio: 3 / 4;
                border-radius: 8px;
                overflow: hidden;
                background: rgba(0, 0, 0, 0.5);
                flex: 0 1 auto;
            `;

            const img = document.createElement("img");
            img.draggable = false;
            installPreviewFallback(img);
            img.src = PLACEHOLDER_IMAGE_PATH;
            img.style.cssText = `
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                object-fit: cover;
                object-position: center center;
                display: block;
                user-select: none;
                pointer-events: none;
                -webkit-user-drag: none;
            `;

            const emptyLabel = document.createElement("div");
            emptyLabel.textContent = "No RefMod selected";
            emptyLabel.style.cssText = `
                position: absolute;
                inset: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                font-size: 11px;
                color: rgba(178, 191, 208, 0.92);
                background: rgba(0, 0, 0, 0.25);
                border-radius: 10px;
                pointer-events: none;
            `;

            portraitFrame.appendChild(img);
            portraitFrame.appendChild(emptyLabel);
            previewBox.appendChild(portraitFrame);

            const clickCatcher = document.createElement("div");
            clickCatcher.style.cssText = `
                position: absolute;
                inset: 0;
                cursor: pointer;
                background: transparent;
            `;
            previewBox.appendChild(clickCatcher);

            previewBox.addEventListener("wheel", (event) => {
                const canvas = app.canvas?.canvas || document.querySelector("canvas.lgraphcanvas");
                if (!canvas) return;
                const forwarded = new WheelEvent("wheel", {
                    bubbles: true,
                    cancelable: true,
                    clientX: event.clientX,
                    clientY: event.clientY,
                    deltaX: event.deltaX,
                    deltaY: event.deltaY,
                    deltaZ: event.deltaZ,
                    deltaMode: event.deltaMode,
                    ctrlKey: event.ctrlKey,
                    shiftKey: event.shiftKey,
                    altKey: event.altKey,
                    metaKey: event.metaKey,
                });
                canvas.dispatchEvent(forwarded);
            }, { passive: true });
            panel.appendChild(previewBox);

            const domWidget = node.addDOMWidget("vrp_preview", "div", panel, {
                serialize: false,
                hideOnZoom: false,
            });
            domWidget.getHeight = () => "100%";

            const strengthWidgetIndex = node.widgets.indexOf(strengthWidget);
            const audioStrengthWidgetIndex = node.widgets.indexOf(audioStrengthWidget);
            const numericStrengthWidget = node.addWidget(
                "number",
                "video_weight",
                strengthWidget?.value != null ? Number(strengthWidget.value) : 1.0,
                (v) => {
                    if (strengthWidget) strengthWidget.value = Number(v);
                    node.setDirtyCanvas(true, true);
                },
                {
                    min: strengthWidget?.options?.min != null ? Number(strengthWidget.options.min) : 0,
                    max: strengthWidget?.options?.max != null ? Number(strengthWidget.options.max) : 10,
                    step: strengthWidget?.options?.step != null ? Number(strengthWidget.options.step) : 0.01,
                    precision: 2,
                }
            );
            numericStrengthWidget.serialize = false;

            const numericAudioStrengthWidget = node.addWidget(
                "number",
                "audio_weight",
                audioStrengthWidget?.value != null ? Number(audioStrengthWidget.value) : 1.0,
                (v) => {
                    if (audioStrengthWidget) audioStrengthWidget.value = Number(v);
                    node.setDirtyCanvas(true, true);
                },
                {
                    min: audioStrengthWidget?.options?.min != null ? Number(audioStrengthWidget.options.min) : 0,
                    max: audioStrengthWidget?.options?.max != null ? Number(audioStrengthWidget.options.max) : 10,
                    step: audioStrengthWidget?.options?.step != null ? Number(audioStrengthWidget.options.step) : 0.01,
                    precision: 2,
                }
            );
            numericAudioStrengthWidget.serialize = false;

            const normalizedStrengthValue = (value, fallback = 1.0) => {
                const parsed = Number(value);
                return Number.isFinite(parsed) ? parsed : fallback;
            };

            const syncNumericStrength = () => {
                const fallback = normalizedStrengthValue(numericStrengthWidget?.value, 1.0);
                const next = normalizedStrengthValue(strengthWidget?.value, fallback);
                numericStrengthWidget.value = next;
                if (strengthWidget) strengthWidget.value = next;
            };

            const syncNumericAudioStrength = () => {
                const fallback = normalizedStrengthValue(numericAudioStrengthWidget?.value, 1.0);
                const next = normalizedStrengthValue(audioStrengthWidget?.value, fallback);
                numericAudioStrengthWidget.value = next;
                if (audioStrengthWidget) audioStrengthWidget.value = next;
            };

            if (strengthWidget) {
                const originalStrengthCb = strengthWidget.callback;
                strengthWidget.callback = function () {
                    syncNumericStrength();
                    if (originalStrengthCb) return originalStrengthCb.apply(this, arguments);
                };
            }
            if (audioStrengthWidget) {
                const originalAudioStrengthCb = audioStrengthWidget.callback;
                audioStrengthWidget.callback = function () {
                    syncNumericAudioStrength();
                    if (originalAudioStrengthCb) return originalAudioStrengthCb.apply(this, arguments);
                };
            }
            syncNumericStrength();
            syncNumericAudioStrength();

            if (strengthWidget) hideWidget(strengthWidget);
            if (audioStrengthWidget) hideWidget(audioStrengthWidget);
            if (modPathWidget) hideWidget(modPathWidget);

            let filePickerWidget = node.addWidget(
                "combo",
                "mod",
                "(none)",
                (label) => {
                    const selected = node._vrpModMap?.[label] || null;
                    setSelection(selected);
                },
                { values: ["(none)"] }
            );
            filePickerWidget.serialize = false;
            node._vrpModMap = { "(none)": null };

            node._vrpStepModByDelta = async (delta) => {
                const direction = delta >= 0 ? 1 : -1;
                const labels = Array.isArray(filePickerWidget?.options?.values)
                    ? filePickerWidget.options.values.filter((label) => label !== "(none)")
                    : [];
                if (labels.length <= 1) return false;

                const currentLabel = String(filePickerWidget?.value || "");
                const currentIndex = labels.indexOf(currentLabel);

                let nextIndex;
                if (currentIndex < 0) {
                    nextIndex = direction > 0 ? 0 : labels.length - 1;
                } else {
                    nextIndex = (currentIndex + direction + labels.length) % labels.length;
                }

                const nextLabel = labels[nextIndex];
                if (!nextLabel) return false;

                filePickerWidget.value = nextLabel;
                if (typeof filePickerWidget.callback === "function") {
                    filePickerWidget.callback(nextLabel);
                } else {
                    const nextEntry = node._vrpModMap?.[nextLabel] || null;
                    setSelection(nextEntry);
                }
                return true;
            };

            installVrpArrowNavigation();

            const loadPreview = async (entry) => {
                if (!entry || !entry.path) {
                    img.removeAttribute("src");
                    img.style.display = "none";
                    emptyLabel.style.display = "flex";
                    return;
                }
                img.style.display = "block";
                emptyLabel.style.display = "none";
                img.src = entry.preview_path ? buildPreviewUrl(entry.preview_path) : PLACEHOLDER_IMAGE_PATH;
            };

            const setSelection = (entry) => {
                const selected = entry && entry.path ? entry : null;
                const value = selected?.path || "";
                if (modPathWidget) modPathWidget.value = value;
                const label = Object.keys(node._vrpModMap || {}).find((key) => node._vrpModMap[key]?.path === value);
                if (filePickerWidget) filePickerWidget.value = label || "(none)";
                node.properties._vrpModPath = value;
                node.properties._vrpModDir = selected?.path ? dirnameForPath(selected.path) : (node.properties._vrpModDir || "");
                node.properties._vrpPreviewPath = selected?.preview_path || "";
                void loadPreview(selected);
                node.setDirtyCanvas(true, true);
            };

            const refreshPickerOptions = async (folderPath, preferredPath = null) => {
                if (!folderPath) return;
                const data = await listBrowserFolder(folderPath);
                const labels = ["(none)"];
                const map = { "(none)": null };
                const used = new Set(labels);
                for (const entry of data.mods || []) {
                    const base = entryDisplayName(entry) || entry.name;
                    let label = base;
                    let idx = 2;
                    while (used.has(label)) {
                        label = `${base} (${idx++})`;
                    }
                    used.add(label);
                    labels.push(label);
                    map[label] = entry;
                }
                node._vrpModMap = map;
                if (filePickerWidget) filePickerWidget.options.values = labels;
                const desiredPath = preferredPath != null ? preferredPath : (modPathWidget?.value || "");
                const desiredLabel = Object.keys(map).find((key) => map[key]?.path === desiredPath);
                if (filePickerWidget) filePickerWidget.value = desiredLabel || "(none)";
                return desiredLabel ? map[desiredLabel] : null;
            };

            const openBrowser = async () => {
                const root = await getRefModsRoot();
                const current = node.properties?._vrpModPath || modPathWidget?.value || "";
                const initialPath = node.properties?._vrpModDir || dirnameForPath(current) || root;
                createRefModBrowserModal(initialPath, async (selected) => {
                    node.properties._vrpModDir = dirnameForPath(selected.path);
                    setSelection(selected);
                    await refreshPickerOptions(node.properties._vrpModDir, selected.path);
                });
            };

            const browseButton = node.addWidget("button", "📁 Select RefMod", null, openBrowser);
            browseButton.serialize = false;

            const modPathWidgetIndex = node.widgets.indexOf(modPathWidget);
            if (modPathWidgetIndex >= 0 && filePickerWidget) {
                const pickerIndex = node.widgets.indexOf(filePickerWidget);
                if (pickerIndex >= 0) {
                    node.widgets.splice(pickerIndex, 1);
                    node.widgets.splice(modPathWidgetIndex + 1, 0, filePickerWidget);
                }
                const buttonIndex = node.widgets.indexOf(browseButton);
                if (buttonIndex >= 0) {
                    node.widgets.splice(buttonIndex, 1);
                    node.widgets.splice(modPathWidgetIndex + 2, 0, browseButton);
                }
                const previewIndex = node.widgets.indexOf(domWidget);
                if (previewIndex >= 0) {
                    node.widgets.splice(previewIndex, 1);
                    node.widgets.splice(modPathWidgetIndex + 3, 0, domWidget);
                }
            }

            if (strengthWidgetIndex >= 0 && numericStrengthWidget) {
                const numericIndex = node.widgets.indexOf(numericStrengthWidget);
                if (numericIndex >= 0) {
                    node.widgets.splice(numericIndex, 1);
                    node.widgets.splice(strengthWidgetIndex + 1, 0, numericStrengthWidget);
                }
            }
            if (audioStrengthWidgetIndex >= 0 && numericAudioStrengthWidget) {
                const numericAudioIndex = node.widgets.indexOf(numericAudioStrengthWidget);
                if (numericAudioIndex >= 0) {
                    node.widgets.splice(numericAudioIndex, 1);
                    node.widgets.splice(audioStrengthWidgetIndex + 1, 0, numericAudioStrengthWidget);
                }
            }

            if (filePickerWidget && browseButton) {
                const pickerIndex = node.widgets.indexOf(filePickerWidget);
                const buttonIndex = node.widgets.indexOf(browseButton);
                if (pickerIndex >= 0 && buttonIndex >= 0) {
                    node.widgets.splice(buttonIndex, 1);
                    node.widgets.splice(pickerIndex + 1, 0, browseButton);
                }
            }

            clickCatcher.addEventListener("click", async (event) => {
                event.stopPropagation();
                await openBrowser();
            });

            const onMouseMove = node.onMouseMove;
            node.onMouseMove = function () {
                _vrpHoveredNode = this;
                return onMouseMove?.apply(this, arguments);
            };

            const onMouseEnter = node.onMouseEnter;
            node.onMouseEnter = function () {
                _vrpHoveredNode = this;
                return onMouseEnter?.apply(this, arguments);
            };

            const onMouseLeave = node.onMouseLeave;
            node.onMouseLeave = function () {
                if (_vrpHoveredNode === this) {
                    _vrpHoveredNode = null;
                }
                return onMouseLeave?.apply(this, arguments);
            };

            if (previewBox) {
                previewBox.addEventListener("mouseenter", () => {
                    _vrpHoveredNode = node;
                });
                previewBox.addEventListener("mouseleave", () => {
                    if (_vrpHoveredNode === node) {
                        _vrpHoveredNode = null;
                    }
                });
            }

            const originalSerialize = node.serialize;
            node.serialize = function () {
                if (strengthWidget && numericStrengthWidget) {
                    strengthWidget.value = normalizedStrengthValue(
                        numericStrengthWidget.value,
                        normalizedStrengthValue(strengthWidget.value, 1.0)
                    );
                    numericStrengthWidget.value = strengthWidget.value;
                }
                if (audioStrengthWidget && numericAudioStrengthWidget) {
                    audioStrengthWidget.value = normalizedStrengthValue(
                        numericAudioStrengthWidget.value,
                        normalizedStrengthValue(audioStrengthWidget.value, 1.0)
                    );
                    numericAudioStrengthWidget.value = audioStrengthWidget.value;
                }
                return originalSerialize ? originalSerialize.apply(this, arguments) : undefined;
            };

            const oldOnResize = node.onResize;
            node.onResize = function (size) {
                const res = oldOnResize?.apply(this, arguments);
                if (size) {
                    if (size[0] < 200) size[0] = 200;
                    if (size[1] < 320) size[1] = 320;
                }
                return res;
            };

            const origDraw = domWidget.draw;
            domWidget.draw = function (ctx, n) {
                if (typeof origDraw === "function") origDraw.apply(this, arguments);
                if (!panel || n.flags?.collapsed) return;
                panel.style.setProperty("width", (n.size[0] - 18) + "px", "important");
                panel.style.setProperty("left", "0px", "important");
                panel.style.setProperty("margin", "0px", "important");
                panel.style.setProperty("padding", "4px", "important");
                panel.style.setProperty("box-sizing", "border-box", "important");
                panel.style.setProperty("overflow", "hidden", "important");
            };

            const onConfigure = node.onConfigure;
            node.onConfigure = async function (info) {
                node.properties._configuredFromWorkflow = true;
                const res = onConfigure?.apply(this, arguments);
                if (info && info.widgets_values) {
                    if (strengthWidget) {
                        const idx = this.widgets.indexOf(strengthWidget);
                        if (idx >= 0 && info.widgets_values[idx] !== undefined && info.widgets_values[idx] !== null) {
                            strengthWidget.value = normalizedStrengthValue(info.widgets_values[idx], 1.0);
                        }
                    }
                    if (audioStrengthWidget) {
                        const idx = this.widgets.indexOf(audioStrengthWidget);
                        if (idx >= 0 && info.widgets_values[idx] !== undefined && info.widgets_values[idx] !== null) {
                            audioStrengthWidget.value = normalizedStrengthValue(info.widgets_values[idx], 1.0);
                        }
                    }
                }
                if (strengthWidget && !Number.isFinite(Number(strengthWidget.value))) {
                    strengthWidget.value = 1.0;
                }
                if (audioStrengthWidget && !Number.isFinite(Number(audioStrengthWidget.value))) {
                    audioStrengthWidget.value = 1.0;
                }
                syncNumericStrength();
                syncNumericAudioStrength();

                const restoredPath = modPathWidget?.value || node.properties?._vrpModPath || "";
                const restoredDir = node.properties?._vrpModDir || dirnameForPath(restoredPath);
                const dir = restoredDir || await getRefModsRoot();
                node.properties._vrpModDir = dir;
                const restoredEntry = await refreshPickerOptions(dir, restoredPath);
                if (restoredEntry) {
                    setSelection(restoredEntry);
                } else {
                    void loadPreview(restoredPath ? { path: restoredPath, preview_path: node.properties?._vrpPreviewPath || "" } : null);
                }
                return res;
            };

            setTimeout(async () => {
                syncNumericStrength();
                syncNumericAudioStrength();
                const initialPath = modPathWidget?.value || node.properties?._vrpModPath || "";
                const dir = node.properties?._vrpModDir || dirnameForPath(initialPath) || await getRefModsRoot();
                node.properties._vrpModDir = dir;
                const initialEntry = await refreshPickerOptions(dir, initialPath || null);
                if (initialEntry) {
                    setSelection(initialEntry);
                } else {
                    void loadPreview(initialPath ? { path: initialPath, preview_path: node.properties?._vrpPreviewPath || "" } : null);
                }
                if (!node.properties?._configuredFromWorkflow) {
                    node.setSize([...DEFAULT_PICKER_NODE_SIZE]);
                }
            }, 10);

            return result;
        };
    },
});

console.log("[H3RefModPicker] VisualRefModPicker extension loaded");