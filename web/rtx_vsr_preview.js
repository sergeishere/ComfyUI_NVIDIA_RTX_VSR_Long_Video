import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function buildViewURL(item) {
    const params = new URLSearchParams();
    params.set("filename", item.filename);
    if (item.subfolder) params.set("subfolder", item.subfolder);
    if (item.type) params.set("type", item.type);
    return api.apiURL(`/view?${params.toString()}`);
}

const MIN_WIDGET_WIDTH = 420;
const MIN_WIDGET_HEIGHT = 280;

// Match the renderer detection and Legacy width fix used by MiniMax H3
// Extender. Vue may briefly re-parent DOM widgets while switching renderers.
function widgetRenderMode(root) {
    if (!root?.isConnected) return "pending";
    const inVueRow = Boolean(root.closest?.(".lg-node-widget"));
    const flag = globalThis.LiteGraph?.vueNodesMode;
    if (typeof flag === "boolean") {
        if (flag !== inVueRow) return "pending";
        return flag ? "nodes2" : "legacy";
    }
    return inVueRow ? "nodes2" : "legacy";
}

function pinLegacyWidgetWidth(state, enabled) {
    const widget = state?.dom;
    if (!widget) return;
    if (enabled) {
        if (state.legacyWidthPinned) return;
        try {
            state.originalWidthDescriptor = Object.getOwnPropertyDescriptor(widget, "width") || null;
            Object.defineProperty(widget, "width", {
                configurable: true,
                enumerable: state.originalWidthDescriptor?.enumerable ?? true,
                get: () => undefined,
                set: () => {},
            });
            state.legacyWidthPinned = true;
        } catch (_) {
            // Best-effort workaround for Legacy DOM widget sidebar sizing.
        }
        return;
    }
    if (!state.legacyWidthPinned) return;
    try {
        if (state.originalWidthDescriptor) {
            Object.defineProperty(widget, "width", state.originalWidthDescriptor);
        } else {
            delete widget.width;
        }
    } catch (_) {}
    state.originalWidthDescriptor = null;
    state.legacyWidthPinned = false;
}

function normalizeNodes2Grid(state) {
    const { root } = state;
    const row = root.closest?.(".lg-node-widget");
    const grid = row?.parentElement?.closest?.(".lg-node-widgets") || row?.parentElement;
    if (!row || !grid?.classList?.contains("lg-node-widgets")) return;

    // Vue may assign every row an 'auto' track. Only the preview may grow;
    // regular controls should remain at their natural height. Otherwise the
    // expanded sidebar / vertical resize can split the node into empty rows.
    const rows = Array.from(grid.children).filter((child) =>
        child?.classList?.contains("col-span-full") &&
        getComputedStyle(child).display !== "none"
    );
    if (!rows.includes(row)) return;
    const tracks = rows.map((child) =>
        child === row ? `minmax(${MIN_WIDGET_HEIGHT}px, 1fr)` : "min-content"
    ).join(" ");
    if (grid.style.gridTemplateRows !== tracks) grid.style.gridTemplateRows = tracks;
    if (grid.style.flex !== "1 1 auto") grid.style.flex = "1 1 auto";
    row.style.minWidth = "0";
    row.style.width = "100%";
    row.style.minHeight = `${MIN_WIDGET_HEIGHT}px`;
    row.style.height = "auto";

    if (state.grid !== grid) {
        state.gridObserver?.disconnect();
        state.grid = grid;
        if (typeof MutationObserver !== "undefined") {
            state.gridObserver = new MutationObserver(() => {
                if (widgetRenderMode(root) === "nodes2" && state.grid?.isConnected) {
                    normalizeNodes2Grid(state);
                }
            });
            state.gridObserver.observe(grid, { attributes: true, attributeFilter: ["style"], childList: true });
        }
    }
}

function attachPreview(node) {
    if (node.__rtxVsrPreview) return;

    const root = document.createElement("div");
    root.dataset.rtxVsrPreview = "";
    root.style.display = "flex";
    root.style.flexDirection = "column";
    root.style.gap = "8px";
    root.style.padding = "8px";
    root.style.boxSizing = "border-box";
    root.style.background = "#111";
    root.style.borderRadius = "8px";
    root.style.width = "100%";
    root.style.maxWidth = "100%";
    root.style.minWidth = "0";
    root.style.height = "320px";
    root.style.minHeight = "0";
    root.style.overflow = "hidden";
    root.style.setProperty("--comfy-widget-min-height", `${MIN_WIDGET_HEIGHT}px`);

    const title = document.createElement("div");
    title.textContent = "Preview";
    title.style.color = "#ddd";
    title.style.fontWeight = "600";
    title.style.fontSize = "13px";

    const video = document.createElement("video");
    video.controls = true;
    video.muted = false;
    video.playsInline = true;
    video.preload = "metadata";
    video.style.width = "100%";
    video.style.flex = "1 1 0";
    video.style.minWidth = "0";
    video.style.minHeight = "0";
    video.style.objectFit = "contain";
    video.style.background = "black";
    video.style.borderRadius = "6px";

    const status = document.createElement("div");
    status.textContent = "No preview yet.";
    status.style.color = "#bbb";
    status.style.fontSize = "12px";
    status.style.whiteSpace = "pre-wrap";

    const link = document.createElement("a");
    link.textContent = "Open output";
    link.target = "_blank";
    link.rel = "noreferrer";
    link.style.color = "#9cc4ff";
    link.style.fontSize = "12px";
    link.hidden = true;

    root.append(title, video, status, link);

    const state = {
        root, video, status, link, dom: null,
        bodyHeight: 320, lastMode: null, legacyWidthPinned: false,
        originalWidthDescriptor: null, grid: null, gridObserver: null,
    };

    const dom = node.addDOMWidget("rtx_vsr_preview", "rtx_vsr_preview", root, {
        serialize: false,
        hideOnZoom: false,
        // Legacy sizes the preview via getHeight; Vue Nodes 2.0 owns its grid
        // track, which must not be driven back from node.size.
        getHeight: () => state.lastMode === "nodes2" ? MIN_WIDGET_HEIGHT : state.bodyHeight + 16,
        getMinHeight: () => MIN_WIDGET_HEIGHT,
        afterResize: () => scheduleLayout(),
    });
    state.dom = dom;
    dom.computeSize = () => [MIN_WIDGET_WIDTH, MIN_WIDGET_HEIGHT];

    function syncLayout(size = node.size, retry = 0) {
        const mode = widgetRenderMode(root);
        if (mode === "pending") {
            if (retry < 12) requestAnimationFrame(() => syncLayout(node.size, retry + 1));
            return;
        }
        state.lastMode = mode;
        pinLegacyWidgetWidth(state, mode === "legacy");
        if (mode === "nodes2") {
            root.style.height = "100%";
            root.style.minHeight = `${MIN_WIDGET_HEIGHT}px`;
            root.style.flex = "1 1 0";
            normalizeNodes2Grid(state);
            return;
        }
        state.gridObserver?.disconnect();
        state.gridObserver = null;
        state.grid = null;
        root.style.flex = "0 0 auto";
        root.style.minHeight = "0";
        const rows = Math.max(node.inputs?.length || 0, node.outputs?.length || 0);
        const top = Number.isFinite(dom?.last_y) && dom.last_y > 0
            ? dom.last_y
            : Number.isFinite(dom?.y) && dom.y > 0 ? dom.y : rows * 20 + 10;
        const nodeHeight = Number.isFinite(size?.[1]) ? size[1] : 520;
        state.bodyHeight = Math.max(MIN_WIDGET_HEIGHT - 16, nodeHeight - top - 20);
        root.style.height = `${state.bodyHeight}px`;
    }

    function scheduleLayout() {
        requestAnimationFrame(() => requestAnimationFrame(() => syncLayout()));
    }

    // Protect the Legacy width as soon as its renderer is known; the getter
    // prevents stale width values while the sidebar is resized/opened/closed.
    syncLayout();

    const oldResize = node.onResize;
    node.onResize = function (size) {
        oldResize?.apply(this, arguments);
        syncLayout(size);
        scheduleLayout();
    };

    const oldExecuted = node.onExecuted;
    node.onExecuted = function (message) {
        oldExecuted?.apply(this, arguments);
        const item = message?.rtx_vsr_preview?.[0] || message?.gifs?.[0];
        if (!item) return;
        const url = buildViewURL(item);
        video.src = url;
        video.load();
        link.href = url;
        link.download = item.filename || "preview.mp4";
        link.hidden = false;
        const report = Array.isArray(message?.rtx_vsr_report) ? message.rtx_vsr_report[0] : "";
        status.textContent = report || `${item.filename || "output"}`;
    };

    const oldConfigure = node.onConfigure;
    node.onConfigure = function () {
        oldConfigure?.apply(this, arguments);
        scheduleLayout();
    };

    const oldDrawForeground = node.onDrawForeground;
    node.onDrawForeground = function () {
        oldDrawForeground?.apply(this, arguments);
        if (widgetRenderMode(root) !== "pending" && widgetRenderMode(root) !== state.lastMode) {
            syncLayout();
        }
    };

    const oldRemoved = node.onRemoved;
    node.onRemoved = function () {
        state.gridObserver?.disconnect();
        pinLegacyWidgetWidth(state, false);
        oldRemoved?.apply(this, arguments);
    };

    node.__rtxVsrPreview = state;
    node.setSize([Math.max(MIN_WIDGET_WIDTH, node.size?.[0] || MIN_WIDGET_WIDTH),
                  Math.max(520, node.size?.[1] || 520)]);
    scheduleLayout();
}

app.registerExtension({
    name: "ComfyUI.NVIDIA.RTXVSRLongVideoPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RTX_VSR_Long_Video") return;
        const old = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = old?.apply(this, arguments);
            attachPreview(this);
            return result;
        };
    },
});
