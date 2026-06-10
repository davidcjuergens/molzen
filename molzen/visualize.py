"""Helpers for visualizing Molzen objects."""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from molzen.constants import HARTREE2EV

if TYPE_CHECKING:
    from molzen.io.molecule import Molecule


_PLOT_FONT_FAMILY = "Arial, Helvetica, sans-serif"


def _frame_count(atom_records: np.ndarray) -> int:
    """Return the number of coordinate frames in atom_records."""
    return atom_records.dtype["coords"].shape[0]


def _coerce_frame_index(atom_records: np.ndarray, frame: int | None) -> int | None:
    """Validate a requested frame index against atom_records."""
    if frame is None:
        return None

    n_frames = _frame_count(atom_records)
    if frame < 0 or frame >= n_frames:
        raise IndexError(f"Frame index {frame} out of range for {n_frames} frame(s).")
    return frame


def _viewer_atom_name(row: np.void, fallback_index: int) -> str:
    """Return a fallback atom name for the viewer export."""
    atom_name = str(row["atom_name"]).strip()
    if atom_name:
        return atom_name[:4]

    element = str(row["element"]).strip().capitalize()
    if element:
        return element[:4]
    return f"A{fallback_index}"[:4]


def _xyz_lines_for_frame(atom_records: np.ndarray, frame_index: int) -> list[str]:
    """Serialize one coordinate frame to XYZ atom lines."""
    lines: list[str] = []
    for i, row in enumerate(atom_records, start=1):
        element = str(row["element"]).strip().capitalize()
        if not element:
            element = _viewer_atom_name(row, i).strip().capitalize() or "X"
        coord = np.asarray(row["coords"][frame_index], dtype=float)
        lines.append(f"{element} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n")
    return lines


def _xyz_text(atom_records: np.ndarray, frame: int | None = None) -> str:
    """Serialize atom_records to XYZ text for py3Dmol."""
    if len(atom_records) == 0:
        raise ValueError("Cannot visualize an empty molecule.")

    frame = _coerce_frame_index(atom_records, frame)
    n_frames = _frame_count(atom_records)
    frame_indices = range(n_frames) if frame is None else (frame,)
    chunks: list[str] = []

    for frame_index in frame_indices:
        chunks.append(f"{len(atom_records)}\n")
        chunks.append(f"Frame {frame_index + 1}\n")
        chunks.extend(_xyz_lines_for_frame(atom_records, frame_index))

    return "".join(chunks)


def _record_float(record: dict[str, Any], key: str) -> float | None:
    """Return a finite float from a record, or None."""
    try:
        value = float(record[key])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _record_int(record: dict[str, Any], key: str) -> int | None:
    """Return an integer from a record, or None."""
    try:
        value = float(record[key])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return int(value)


def _state_energy_label(multiplicity: Any, state_j: int | None) -> str:
    """Return a compact state label for an excited-state energy trace."""
    if state_j is None:
        return "state"
    if isinstance(multiplicity, str) and multiplicity:
        prefix = multiplicity.strip()[:1].upper()
        if prefix:
            return f"{prefix}{state_j}"
    return f"state {state_j}"


def _state_transition_label(
    multiplicity: Any, state_i: int | None, state_j: int | None
) -> str:
    """Return a compact transition label for oscillator-strength traces."""
    if state_i is None or state_j is None:
        return "transition"
    if isinstance(multiplicity, str) and multiplicity:
        prefix = multiplicity.strip()[:1].upper()
        if prefix:
            return f"{prefix}{state_i}->{prefix}{state_j}"
    return f"{state_i}->{state_j}"


def _first_frame_s0_energy_ev(records: list[dict[str, Any]]) -> float | None:
    """Return the first-frame S0 total energy in eV, if present."""
    for record in records:
        frame_index = _record_int(record, "frame_index")
        state_j = _record_int(record, "state_j")
        energy = _record_float(record, "total_energy_au")
        if frame_index == 0 and state_j == 0 and energy is not None:
            return energy * HARTREE2EV
    return None


def _excited_state_energy_series(
    records: list[dict[str, Any]] | None,
    *,
    energy_key: str = "total_energy_au",
    energy_scale: float = HARTREE2EV,
) -> list[dict[str, Any]]:
    """Convert excited-state records into S0-frame-relative eV traces."""
    if not records:
        return []

    baseline_ev = _first_frame_s0_energy_ev(records)
    grouped: dict[tuple[str, int | None], dict[int, float]] = {}
    labels: dict[tuple[str, int | None], str] = {}
    for record in records:
        frame_index = _record_int(record, "frame_index")
        energy = _record_float(record, energy_key)
        if frame_index is None or energy is None:
            continue
        energy *= energy_scale
        if baseline_ev is not None:
            energy -= baseline_ev

        multiplicity = record.get("multiplicity") or ""
        state_j = _record_int(record, "state_j")
        key = (str(multiplicity), state_j)
        grouped.setdefault(key, {})
        grouped[key].setdefault(frame_index, energy)
        labels[key] = _state_energy_label(multiplicity, state_j)

    series = []
    for key, frame_to_energy in grouped.items():
        frames = sorted(frame_to_energy)
        series.append(
            {
                "label": labels[key],
                "x": frames,
                "y": [frame_to_energy[frame] for frame in frames],
            }
        )
    return sorted(series, key=lambda item: item["label"])


def _excited_state_oscillator_strength_series(
    records: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Convert S0 -> Sn oscillator strengths into frame-indexed traces."""
    if not records:
        return []

    grouped: dict[tuple[str, int | None, int | None], dict[int, float]] = {}
    labels: dict[tuple[str, int | None, int | None], str] = {}
    for record in records:
        frame_index = _record_int(record, "frame_index")
        osc_strength = _record_float(record, "osc_strength")
        state_i = _record_int(record, "state_i")
        state_j = _record_int(record, "state_j")
        if (
            frame_index is None
            or osc_strength is None
            or state_i != 0
            or state_j in (None, 0)
        ):
            continue

        multiplicity = record.get("multiplicity") or ""
        key = (str(multiplicity), state_i, state_j)
        grouped.setdefault(key, {})
        grouped[key].setdefault(frame_index, osc_strength)
        labels[key] = _state_transition_label(multiplicity, state_i, state_j)

    series = []
    for key, frame_to_osc in grouped.items():
        frames = sorted(frame_to_osc)
        series.append(
            {
                "label": labels[key],
                "x": frames,
                "y": [frame_to_osc[frame] for frame in frames],
                "color_index": key[2],
            }
        )
    return sorted(series, key=lambda item: item["label"])


def _pixel_size(value: str | int, default: int) -> int:
    """Return an integer pixel size for simple px values."""
    if isinstance(value, int):
        return value
    value = value.strip()
    if value.endswith("px"):
        try:
            return int(float(value[:-2]))
        except ValueError:
            return default
    return default


def _require_py3dmol() -> Any:
    """Import py3Dmol lazily for optional visualization support."""
    try:
        import py3Dmol
    except ImportError as exc:
        raise ImportError(
            "py3Dmol is required for show_molecule_py3dmol(). "
            "Install py3Dmol to use molecule visualization. "
            f"Original import error: {exc}"
        ) from exc
    return py3Dmol


def _center_py3dmol_view(view: Any) -> None:
    """Center the py3Dmol viewer div in notebook output."""
    view.startjs = view.startjs.replace(
        'style="position: relative; width:',
        'style="position: relative; margin-left: auto; margin-right: auto; width:',
        1,
    )


def _coerce_hover_duration_ms(hover_duration: float) -> int:
    """Return a finite, non-negative hover duration in milliseconds."""
    try:
        duration = float(hover_duration)
    except (TypeError, ValueError) as exc:
        raise ValueError("atom_hover_duration must be a number of seconds.") from exc
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("atom_hover_duration must be a finite, non-negative number.")
    return int(round(duration * 1000))


def _set_py3dmol_hover_duration(view: Any, hover_duration: float) -> None:
    """Configure 3Dmol's hover callback delay on a py3Dmol view."""
    hover_duration_ms = _coerce_hover_duration_ms(hover_duration)
    viewer_config = '{backgroundColor:"white"}'
    if viewer_config not in view.startjs:
        raise ValueError("Could not find py3Dmol viewer config for hover duration.")
    view.startjs = view.startjs.replace(
        viewer_config,
        f'{{backgroundColor:"white", hoverDuration: {hover_duration_ms}}}',
        1,
    )


def _insert_py3dmol_html_before_script(view: Any, html_fragment: str) -> None:
    """Insert HTML between the py3Dmol viewer markup and startup script."""
    script_marker = "<script>\n"
    if script_marker not in view.startjs:
        raise ValueError("Could not find py3Dmol script marker for HTML injection.")
    view.startjs = view.startjs.replace(
        script_marker, f"{html_fragment}{script_marker}", 1
    )


def _add_py3dmol_frame_slider(view: Any, n_frames: int, width: str | int) -> None:
    """Attach a simple 3Dmol.js frame slider to a py3Dmol view."""
    width_css = f"{width}px" if isinstance(width, int) else width
    slider_html = f"""
<div id="3dmol_frame_controls_UNIQUEID" style="display: flex; align-items: center; gap: 8px; width: {width_css}; margin: 4px auto 0 auto; font: 12px sans-serif;">
  <input id="3dmol_frame_slider_UNIQUEID" type="range" min="0" max="{n_frames - 1}" value="0" step="1" style="flex: 1;">
  <span>Frame <span id="3dmol_frame_label_UNIQUEID">1</span>/{n_frames}</span>
</div>
"""
    script = """
var frameSlider_UNIQUEID = document.getElementById("3dmol_frame_slider_UNIQUEID");
var frameLabel_UNIQUEID = document.getElementById("3dmol_frame_label_UNIQUEID");
if(frameSlider_UNIQUEID && frameLabel_UNIQUEID) {
    frameSlider_UNIQUEID.addEventListener("input", function() {
        var frame = parseInt(this.value);
        frameLabel_UNIQUEID.textContent = String(frame + 1);
        if(typeof window.molzenUpdateEnergyFrame_UNIQUEID === "function") {
            window.molzenUpdateEnergyFrame_UNIQUEID(frame);
        }
        if(typeof window.molzenClearAtomHoverLabel_UNIQUEID === "function") {
            window.molzenClearAtomHoverLabel_UNIQUEID();
        }
        function finishFrameUpdate() {
            if(typeof window.molzenEnableAtomHoverLabels_UNIQUEID === "function") {
                window.molzenEnableAtomHoverLabels_UNIQUEID();
            }
            viewer_UNIQUEID.render();
        }
        var framePromise = viewer_UNIQUEID.setFrame(frame);
        if(framePromise && typeof framePromise.then === "function") {
            framePromise.then(finishFrameUpdate);
        } else {
            finishFrameUpdate();
        }
    });
}
"""
    _insert_py3dmol_html_before_script(view, slider_html)
    view.startjs += script


def _gif_export_frame_count(n_frames: int, bounce: bool) -> int:
    """Return the number of frames that will be encoded into the GIF."""
    return n_frames + (n_frames - 1 if bounce and n_frames > 1 else 0)


def _gif_total_time_delays_ms(
    n_frames: int,
    *,
    bounce: bool,
    total_time: float,
) -> list[int]:
    """Return per-frame GIF delays in ms that sum to total_time centiseconds."""
    total_time = float(total_time)
    if not np.isfinite(total_time) or total_time <= 0:
        raise ValueError("gif_total_time must be a positive number of seconds.")

    frame_count = _gif_export_frame_count(n_frames, bounce)
    target_centiseconds = int(np.floor(total_time * 100 + 0.5))
    if target_centiseconds < frame_count:
        minimum_time = frame_count / 100
        raise ValueError(
            "gif_total_time is too short for the exported frame count at GIF "
            f"centisecond resolution; use at least {minimum_time:.2f} seconds."
        )

    delays_cs: list[int] = []
    previous_cumulative = 0
    half_frame_count = frame_count // 2
    for i in range(1, frame_count + 1):
        cumulative = (i * target_centiseconds + half_frame_count) // frame_count
        delays_cs.append(cumulative - previous_cumulative)
        previous_cumulative = cumulative

    return [delay_cs * 10 for delay_cs in delays_cs]


def _integer_y_ticks(
    y_min: float,
    y_max: float,
    *,
    target_intervals: int = 4,
) -> list[int]:
    """Return evenly spaced integer y ticks inside the visible y range."""
    if target_intervals <= 0:
        raise ValueError("target_intervals must be positive.")

    y_min = float(y_min)
    y_max = float(y_max)
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        raise ValueError("y tick values must be finite.")
    if y_max <= y_min:
        raise ValueError("y_max must be greater than y_min.")

    span = y_max - y_min
    increment = max(1, int(np.ceil(span / target_intervals)))
    first_tick = increment * int(np.ceil(y_min / increment))
    last_tick = increment * int(np.floor(y_max / increment))
    if first_tick > last_tick:
        return [int(round((y_min + y_max) / 2))]
    return list(range(first_tick, last_tick + increment, increment))


def _fixed_y_ticks(y_min: float, y_max: float, increment: float) -> list[float]:
    """Return fixed-increment y ticks inside the visible y range."""
    y_min = float(y_min)
    y_max = float(y_max)
    increment = float(increment)
    if not np.isfinite(y_min) or not np.isfinite(y_max) or not np.isfinite(increment):
        raise ValueError("y tick values must be finite.")
    if y_max <= y_min:
        raise ValueError("y_max must be greater than y_min.")
    if increment <= 0:
        raise ValueError("increment must be positive.")

    first_tick_index = int(np.ceil(y_min / increment))
    last_tick_index = int(np.floor(y_max / increment))
    return [
        round(i * increment, 10) for i in range(first_tick_index, last_tick_index + 1)
    ]


def _tick_label(value: float | int) -> str:
    """Return a compact tick label without unnecessary trailing zeros."""
    if np.isclose(float(value), 0.0):
        return "0"
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _add_py3dmol_gif_export_controls(
    view: Any,
    n_frames: int,
    width: str | int,
    *,
    delay_ms: int = 120,
    bounce: bool = False,
    total_time: float | None = None,
    png_scale: float = 2.0,
    filename: str = "molzen.gif",
    png_filename: str = "molzen.png",
) -> None:
    """Attach browser-side GIF export controls to a py3Dmol trajectory view."""
    if total_time is not None:
        frame_delays_ms = _gif_total_time_delays_ms(
            n_frames,
            bounce=bounce,
            total_time=total_time,
        )
        delay_ms = frame_delays_ms[0]
    else:
        delay_ms = int(delay_ms)
        if delay_ms <= 0:
            raise ValueError("gif_delay_ms must be a positive integer.")
        frame_delays_ms = []

    png_scale = float(png_scale)
    if not np.isfinite(png_scale) or png_scale <= 0:
        raise ValueError("png_scale must be a positive number.")

    width_css = f"{width}px" if isinstance(width, int) else width
    bounce_json = json.dumps(bool(bounce))
    frame_delays_ms_json = json.dumps(frame_delays_ms)
    png_scale_json = json.dumps(png_scale)
    filename_json = json.dumps(filename)
    png_filename_json = json.dumps(png_filename)
    controls_html = f"""
<div id="molzen_gif_export_controls_UNIQUEID" style="width: {width_css}; min-width: max-content; margin: 6px auto 0 auto; font: 12px {_PLOT_FONT_FAMILY}; text-align: center;">
  <div id="molzen_export_button_group_UNIQUEID" style="position: relative; display: inline-flex; align-items: center; justify-content: center; gap: 8px;">
    <button id="molzen_clipboard_copy_button_UNIQUEID" type="button" style="font: inherit; padding: 4px 10px; cursor: pointer; white-space: nowrap; flex: 0 0 auto;">Copy PNG</button>
    <button id="molzen_png_export_button_UNIQUEID" type="button" style="font: inherit; padding: 4px 10px; cursor: pointer; white-space: nowrap; flex: 0 0 auto;">Export PNG</button>
    <button id="molzen_gif_clipboard_copy_button_UNIQUEID" type="button" style="font: inherit; padding: 4px 10px; cursor: pointer; white-space: nowrap; flex: 0 0 auto;">Copy GIF</button>
    <button id="molzen_gif_export_button_UNIQUEID" type="button" style="font: inherit; padding: 4px 10px; cursor: pointer; white-space: nowrap; flex: 0 0 auto;">Export GIF</button>
    <span id="molzen_gif_export_status_UNIQUEID" style="position: absolute; left: calc(100% + 8px); top: 50%; transform: translateY(-50%); color: #555; display: inline-block; width: 220px; white-space: nowrap; text-align: left;"></span>
  </div>
</div>
"""
    script = f"""
(function() {{
    var copyClipboardButton = document.getElementById("molzen_clipboard_copy_button_UNIQUEID");
    var pngExportButton = document.getElementById("molzen_png_export_button_UNIQUEID");
    var copyGifButton = document.getElementById("molzen_gif_clipboard_copy_button_UNIQUEID");
    var exportButton = document.getElementById("molzen_gif_export_button_UNIQUEID");
    var exportStatus = document.getElementById("molzen_gif_export_status_UNIQUEID");
    if(
        !copyClipboardButton ||
        !pngExportButton ||
        !copyGifButton ||
        !exportButton ||
        !exportStatus
    ) {{
        return;
    }}

    var gifLibraryPromise = null;
    var gifWorkerScriptUrlPromise = null;
    var gifWorkerScriptUrl = "https://cdn.jsdelivr.net/npm/gif.js@0.2.0/dist/gif.worker.js";
    var frameDelaysMs = {frame_delays_ms_json};
    var pngScale = {png_scale_json};
    function setStatus(message) {{
        exportStatus.textContent = message;
    }}

    function setExportButtonsDisabled(disabled) {{
        copyClipboardButton.disabled = disabled;
        pngExportButton.disabled = disabled;
        copyGifButton.disabled = disabled;
        exportButton.disabled = disabled;
    }}

    function loadGifLibrary() {{
        if(typeof window.GIF === "function") {{
            return Promise.resolve();
        }}
        if(gifLibraryPromise) {{
            return gifLibraryPromise;
        }}
        gifLibraryPromise = new Promise(function(resolve, reject) {{
            var gifAmdFactory = null;
            var previousDefine = window.define;
            var previousExports = window.exports;
            var previousModule = window.module;
            if(typeof previousDefine === "function" && previousDefine.amd) {{
                window.define = function(dependencies, factory) {{
                    if(
                        Array.isArray(dependencies) &&
                        dependencies.length === 0 &&
                        typeof factory === "function"
                    ) {{
                        gifAmdFactory = factory;
                        return;
                    }}
                    return previousDefine.apply(this, arguments);
                }};
                window.define.amd = previousDefine.amd;
            }} else {{
                window.define = undefined;
            }}
            window.exports = undefined;
            window.module = undefined;

            function restoreModuleLoaderGlobals() {{
                window.define = previousDefine;
                window.exports = previousExports;
                window.module = previousModule;
            }}

            function findGifConstructor() {{
                if(typeof window.GIF === "function") {{
                    return window.GIF;
                }}
                try {{
                    if(typeof GIF === "function") {{
                        return GIF;
                    }}
                }} catch(error) {{}}
                if(typeof gifAmdFactory === "function") {{
                    try {{
                        return gifAmdFactory();
                    }} catch(error) {{}}
                }}
                try {{
                    if(typeof module !== "undefined" && module) {{
                        if(typeof module.exports === "function") {{
                            return module.exports;
                        }}
                        if(
                            module.exports &&
                            typeof module.exports.GIF === "function"
                        ) {{
                            return module.exports.GIF;
                        }}
                    }}
                }} catch(error) {{}}
                try {{
                    if(typeof exports === "function") {{
                        return exports;
                    }}
                    if(exports && typeof exports.GIF === "function") {{
                        return exports.GIF;
                    }}
                }} catch(error) {{}}
                if(typeof window.exports === "function") {{
                    return window.exports;
                }}
                if(window.exports && typeof window.exports.GIF === "function") {{
                    return window.exports.GIF;
                }}
                if(window.module && typeof window.module.exports === "function") {{
                    return window.module.exports;
                }}
                if(
                    window.module &&
                    window.module.exports &&
                    typeof window.module.exports.GIF === "function"
                ) {{
                    return window.module.exports.GIF;
                }}
                return null;
            }}

            var script = document.createElement("script");
            script.src = "https://cdn.jsdelivr.net/npm/gif.js@0.2.0/dist/gif.min.js";
            script.async = true;
            script.onload = function() {{
                var gifConstructor = findGifConstructor();
                restoreModuleLoaderGlobals();
                if(typeof gifConstructor === "function") {{
                    window.GIF = gifConstructor;
                    resolve();
                }} else {{
                    reject(new Error("GIF encoder loaded but was not available."));
                }}
            }};
            script.onerror = function() {{
                restoreModuleLoaderGlobals();
                reject(new Error("Could not load gif.js encoder."));
            }};
            document.head.appendChild(script);
        }});
        return gifLibraryPromise;
    }}

    function loadGifWorkerScriptUrl() {{
        if(gifWorkerScriptUrlPromise) {{
            return gifWorkerScriptUrlPromise;
        }}
        if(
            typeof window.fetch !== "function" ||
            typeof window.Blob !== "function" ||
            !window.URL ||
            typeof window.URL.createObjectURL !== "function"
        ) {{
            gifWorkerScriptUrlPromise = Promise.resolve(gifWorkerScriptUrl);
            return gifWorkerScriptUrlPromise;
        }}
        gifWorkerScriptUrlPromise = window.fetch(gifWorkerScriptUrl).then(
            function(response) {{
                if(!response.ok) {{
                    throw new Error("Could not load gif.js worker.");
                }}
                return response.text();
            }}
        ).then(function(workerSource) {{
            return window.URL.createObjectURL(
                new window.Blob([workerSource], {{type: "application/javascript"}})
            );
        }}).catch(function() {{
            return gifWorkerScriptUrl;
        }});
        return gifWorkerScriptUrlPromise;
    }}

    function waitForPaint() {{
        return new Promise(function(resolve) {{
            window.requestAnimationFrame(function() {{
                window.requestAnimationFrame(resolve);
            }});
        }});
    }}

    function loadImage(src) {{
        return new Promise(function(resolve, reject) {{
            var image = new Image();
            image.onload = function() {{ resolve(image); }};
            image.onerror = reject;
            image.src = src;
        }});
    }}

    function svgImagePromise(svgElement) {{
        if(!svgElement) {{
            return Promise.resolve(null);
        }}
        var clone = svgElement.cloneNode(true);
        var serialized = new XMLSerializer().serializeToString(clone);
        var encoded = window.btoa(unescape(encodeURIComponent(serialized)));
        return loadImage("data:image/svg+xml;base64," + encoded);
    }}

    function renderedRect(element) {{
        if(!element || typeof element.getBoundingClientRect !== "function") {{
            return null;
        }}
        var rect = element.getBoundingClientRect();
        if(rect.width <= 0 || rect.height <= 0) {{
            return null;
        }}
        return rect;
    }}

    function normalizedScale(scale) {{
        scale = Number(scale);
        if(!Number.isFinite(scale) || scale <= 0) {{
            return 1;
        }}
        return scale;
    }}

    function makeCanvas(width, height, scale) {{
        scale = normalizedScale(scale);
        var canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.ceil(width * scale));
        canvas.height = Math.max(1, Math.ceil(height * scale));
        return canvas;
    }}

    function fillWhite(context, canvas) {{
        context.fillStyle = "white";
        context.fillRect(0, 0, canvas.width, canvas.height);
    }}

    function drawPanelLabels(context, panelElement, panelRect) {{
        var labelSpans = panelElement.querySelectorAll("div span");
        for(var i = 0; i < labelSpans.length; i++) {{
            var label = labelSpans[i];
            var labelText = label.textContent || "";
            var labelRect = renderedRect(label);
            if(!labelText || !labelRect) {{
                continue;
            }}
            var style = window.getComputedStyle
                ? window.getComputedStyle(label)
                : null;
            context.font = style && style.font
                ? style.font
                : "12px Arial, Helvetica, sans-serif";
            context.fillStyle = style && style.color ? style.color : "#222";
            context.textBaseline = "top";
            context.fillText(
                labelText,
                labelRect.left - panelRect.left,
                labelRect.top - panelRect.top
            );
        }}
    }}

    function captureMoleculePanel(scale) {{
        scale = normalizedScale(scale);
        var viewerElement = document.getElementById("3dmolviewer_UNIQUEID");
        var viewerRect = renderedRect(viewerElement);
        return loadImage(viewer_UNIQUEID.pngURI()).then(function(image) {{
            var width = viewerRect ? viewerRect.width : image.width;
            var height = viewerRect ? viewerRect.height : image.height;
            var canvas = makeCanvas(width, height, scale);
            var context = canvas.getContext("2d");
            fillWhite(context, canvas);
            context.drawImage(image, 0, 0, canvas.width, canvas.height);
            return {{
                canvas: canvas,
                rect: viewerRect || {{
                    left: 0,
                    top: 0,
                    right: width,
                    bottom: height,
                    width: width,
                    height: height
                }}
            }};
        }});
    }}

    function capturePlotPanel(panelId, scale) {{
        scale = normalizedScale(scale);
        var panelElement = document.getElementById(
            "molzen_" + panelId + "_panel_UNIQUEID"
        );
        var svgElement = document.getElementById(
            "molzen_" + panelId + "_svg_UNIQUEID"
        );
        var panelRect = renderedRect(panelElement);
        var svgRect = renderedRect(svgElement);
        if(!panelElement || !svgElement || !panelRect || !svgRect) {{
            return Promise.resolve(null);
        }}

        return svgImagePromise(svgElement).then(function(svgImage) {{
            var canvas = makeCanvas(panelRect.width, panelRect.height, scale);
            var context = canvas.getContext("2d");
            fillWhite(context, canvas);
            context.save();
            context.scale(scale, scale);
            context.drawImage(
                svgImage,
                svgRect.left - panelRect.left,
                svgRect.top - panelRect.top,
                svgRect.width,
                svgRect.height
            );
            drawPanelLabels(context, panelElement, panelRect);
            context.restore();
            return {{canvas: canvas, rect: panelRect}};
        }});
    }}

    function composePositionedCaptures(captures, rowRect, scale) {{
        scale = normalizedScale(scale);
        var minLeft = rowRect.left;
        var minTop = rowRect.top;
        var maxRight = rowRect.right;
        var maxBottom = rowRect.bottom;
        for(var i = 0; i < captures.length; i++) {{
            minLeft = Math.min(minLeft, captures[i].rect.left);
            minTop = Math.min(minTop, captures[i].rect.top);
            maxRight = Math.max(maxRight, captures[i].rect.right);
            maxBottom = Math.max(maxBottom, captures[i].rect.bottom);
        }}

        var canvas = makeCanvas(maxRight - minLeft, maxBottom - minTop, scale);
        var context = canvas.getContext("2d");
        fillWhite(context, canvas);
        context.save();
        context.scale(scale, scale);
        for(var j = 0; j < captures.length; j++) {{
            context.drawImage(
                captures[j].canvas,
                captures[j].rect.left - minLeft,
                captures[j].rect.top - minTop,
                captures[j].rect.width,
                captures[j].rect.height
            );
        }}
        context.restore();
        return canvas;
    }}

    function composeFallbackCaptures(captures, scale) {{
        scale = normalizedScale(scale);
        if(captures.length === 1) {{
            return captures[0].canvas;
        }}

        var gap = 12;
        var canvasWidth = 0;
        var canvasHeight = 0;
        for(var i = 0; i < captures.length; i++) {{
            canvasWidth += captures[i].rect.width;
            if(i > 0) {{
                canvasWidth += gap;
            }}
            canvasHeight = Math.max(canvasHeight, captures[i].rect.height);
        }}

        var canvas = makeCanvas(canvasWidth, canvasHeight, scale);
        var context = canvas.getContext("2d");
        fillWhite(context, canvas);
        context.save();
        context.scale(scale, scale);
        var x = 0;
        for(var j = 0; j < captures.length; j++) {{
            context.drawImage(
                captures[j].canvas,
                x,
                0,
                captures[j].rect.width,
                captures[j].rect.height
            );
            x += captures[j].rect.width + gap;
        }}
        context.restore();
        return canvas;
    }}

    function composeFrameCaptures(captures, scale) {{
        scale = normalizedScale(scale);
        captures = captures.filter(function(capture) {{
            return capture !== null;
        }});
        if(captures.length === 0) {{
            throw new Error("No panels were captured.");
        }}

        var rowRect = renderedRect(
            document.getElementById("molzen_show_row_UNIQUEID")
        );
        if(rowRect) {{
            return composePositionedCaptures(captures, rowRect, scale);
        }}
        return composeFallbackCaptures(captures, scale);
    }}

    function captureFrame(scale) {{
        scale = normalizedScale(scale);
        return Promise.all([
            captureMoleculePanel(scale),
            capturePlotPanel("energy", scale),
            capturePlotPanel("oscillator", scale)
        ]).then(function(captures) {{
            return composeFrameCaptures(captures, scale);
        }});
    }}

    function buildExportFrameSequence() {{
        var frames = [];
        for(var i = 0; i < {n_frames}; i++) {{
            frames.push(i);
        }}
        if({bounce_json} && {n_frames} > 1) {{
            for(var j = {n_frames} - 2; j >= 0; j--) {{
                frames.push(j);
            }}
        }}
        return frames;
    }}

    function currentFrame() {{
        var slider = document.getElementById("3dmol_frame_slider_UNIQUEID");
        if(!slider) {{
            return 0;
        }}
        var frame = parseInt(slider.value, 10);
        if(!Number.isFinite(frame)) {{
            return 0;
        }}
        return Math.max(0, Math.min({n_frames - 1}, frame));
    }}

    function setFrame(frame) {{
        var slider = document.getElementById("3dmol_frame_slider_UNIQUEID");
        var label = document.getElementById("3dmol_frame_label_UNIQUEID");
        if(slider) {{
            slider.value = String(frame);
        }}
        if(label) {{
            label.textContent = String(frame + 1);
        }}
        if(typeof window.molzenUpdateEnergyFrame_UNIQUEID === "function") {{
            window.molzenUpdateEnergyFrame_UNIQUEID(frame);
        }}
        if(typeof window.molzenClearAtomHoverLabel_UNIQUEID === "function") {{
            window.molzenClearAtomHoverLabel_UNIQUEID();
        }}
        function finishFrameUpdate() {{
            if(typeof window.molzenEnableAtomHoverLabels_UNIQUEID === "function") {{
                window.molzenEnableAtomHoverLabels_UNIQUEID();
            }}
            viewer_UNIQUEID.render();
            return waitForPaint();
        }}
        var framePromise = viewer_UNIQUEID.setFrame(frame);
        if(framePromise && typeof framePromise.then === "function") {{
            return framePromise.then(finishFrameUpdate);
        }}
        return finishFrameUpdate();
    }}

    function downloadBlob(blob, filename) {{
        var link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        setTimeout(function() {{
            URL.revokeObjectURL(link.href);
            document.body.removeChild(link);
        }}, 0);
    }}

    function dataUrlToBlob(dataUrl) {{
        var parts = dataUrl.split(",");
        var metadata = parts[0] || "";
        var match = metadata.match(/data:([^;]+);base64/);
        var mimeType = match ? match[1] : "image/png";
        var binary = window.atob(parts[1]);
        var bytes = new Uint8Array(binary.length);
        for(var i = 0; i < binary.length; i++) {{
            bytes[i] = binary.charCodeAt(i);
        }}
        return new window.Blob([bytes], {{type: mimeType}});
    }}

    function canvasToPngBlob(canvas) {{
        return new Promise(function(resolve, reject) {{
            if(typeof canvas.toBlob === "function") {{
                canvas.toBlob(function(blob) {{
                    if(!blob) {{
                        reject(new Error("Could not create PNG."));
                        return;
                    }}
                    resolve(blob);
                }}, "image/png");
                return;
            }}

            try {{
                resolve(dataUrlToBlob(canvas.toDataURL("image/png")));
            }} catch(error) {{
                reject(error);
            }}
        }});
    }}

    function downloadCanvasPng(canvas, filename) {{
        return canvasToPngBlob(canvas).then(function(blob) {{
            downloadBlob(blob, filename);
        }});
    }}

    function copyBlobToClipboard(blob, mimeType, unavailableMessage) {{
        if(
            !navigator.clipboard ||
            typeof navigator.clipboard.write !== "function" ||
            typeof window.ClipboardItem !== "function"
        ) {{
            return Promise.reject(new Error(unavailableMessage));
        }}
        if(
            typeof window.ClipboardItem.supports === "function" &&
            !window.ClipboardItem.supports(mimeType)
        ) {{
            return Promise.reject(new Error(unavailableMessage));
        }}

        var clipboardPayload = {{}};
        clipboardPayload[mimeType] = blob;
        var item = new window.ClipboardItem(clipboardPayload);
        return navigator.clipboard.write([item]);
    }}

    function copyCanvasPngToClipboard(canvas) {{
        return canvasToPngBlob(canvas).then(function(blob) {{
            return copyBlobToClipboard(
                blob,
                "image/png",
                "Clipboard image copy is not available in this browser."
            );
        }});
    }}

    function copyGifBlobToClipboard(blob) {{
        return copyBlobToClipboard(
            blob,
            "image/gif",
            "Clipboard GIF copy is not available in this browser."
        );
    }}

    function renderGifBlob() {{
        return Promise.all([loadGifLibrary(), loadGifWorkerScriptUrl()]).then(
            function(values) {{
                var activeWorkerScriptUrl = values[1];
                var captures = [];
                var frameSequence = buildExportFrameSequence();
                var sequenceIndex = 0;

                function captureNextFrame() {{
                    if(sequenceIndex >= frameSequence.length) {{
                        return captures;
                    }}
                    var frame = frameSequence[sequenceIndex];
                    setStatus(
                        "Capturing frame " +
                        String(sequenceIndex + 1) +
                        "/" +
                        String(frameSequence.length) +
                        "..."
                    );
                    return setFrame(frame).then(function() {{
                        return captureFrame();
                    }}).then(function(canvas) {{
                        captures.push(canvas);
                        sequenceIndex += 1;
                        return captureNextFrame();
                    }});
                }}

                return captureNextFrame().then(function() {{
                    return activeWorkerScriptUrl;
                }}).then(function(workerScriptUrl) {{
                    if(captures.length === 0) {{
                        throw new Error("No frames were captured.");
                    }}
                    setStatus("Encoding GIF...");
                    return new Promise(function(resolve, reject) {{
                        var gif = new window.GIF({{
                            workers: 2,
                            quality: 10,
                            width: captures[0].width,
                            height: captures[0].height,
                            workerScript: workerScriptUrl
                        }});
                        for(var i = 0; i < captures.length; i++) {{
                            var frameDelay = i < frameDelaysMs.length
                                ? frameDelaysMs[i]
                                : {delay_ms};
                            gif.addFrame(captures[i], {{
                                copy: true,
                                delay: frameDelay
                            }});
                        }}
                        gif.on("finished", resolve);
                        gif.on("abort", function() {{
                            reject(new Error("GIF encoding was aborted."));
                        }});
                        try {{
                            gif.render();
                        }} catch(error) {{
                            reject(error);
                        }}
                    }});
                }});
            }}
        );
    }}

    copyClipboardButton.addEventListener("click", function() {{
        setExportButtonsDisabled(true);
        setStatus("Copying PNG...");
        setFrame(currentFrame()).then(function() {{
            return captureFrame(pngScale);
        }}).then(function(canvas) {{
            return copyCanvasPngToClipboard(canvas);
        }}).then(function() {{
            setStatus("Copied PNG to clipboard.");
            setExportButtonsDisabled(false);
        }}).catch(function(error) {{
            setStatus(error && error.message ? error.message : "PNG copy failed.");
            setExportButtonsDisabled(false);
        }});
    }});

    pngExportButton.addEventListener("click", function() {{
        setExportButtonsDisabled(true);
        setStatus("Capturing PNG...");
        setFrame(currentFrame()).then(function() {{
            return captureFrame(pngScale);
        }}).then(function(canvas) {{
            return downloadCanvasPng(canvas, {png_filename_json});
        }}).then(function() {{
            setStatus("Downloaded " + {png_filename_json});
            setExportButtonsDisabled(false);
        }}).catch(function(error) {{
            setStatus(error && error.message ? error.message : "PNG export failed.");
            setExportButtonsDisabled(false);
        }});
    }});

    copyGifButton.addEventListener("click", function() {{
        setExportButtonsDisabled(true);
        setStatus("Loading encoder...");
        renderGifBlob().then(function(blob) {{
            setStatus("Copying GIF...");
            return copyGifBlobToClipboard(blob);
        }}).then(function() {{
            setStatus("Copied GIF to clipboard.");
            setExportButtonsDisabled(false);
        }}).catch(function(error) {{
            setStatus(error && error.message ? error.message : "GIF copy failed.");
            setExportButtonsDisabled(false);
        }});
    }});

    exportButton.addEventListener("click", function() {{
        setExportButtonsDisabled(true);
        setStatus("Loading encoder...");
        renderGifBlob().then(function(blob) {{
            downloadBlob(blob, {filename_json});
            setStatus("Downloaded " + {filename_json});
            setExportButtonsDisabled(false);
        }}).catch(function(error) {{
            setStatus(error && error.message ? error.message : "GIF export failed.");
            setExportButtonsDisabled(false);
        }});
    }});
}})();
"""
    _insert_py3dmol_html_before_script(view, controls_html)
    view.startjs += script


def _state_trace_plot_html(
    panel_id: str,
    title: str,
    series: list[dict[str, Any]],
    *,
    n_frames: int,
    frame: int | None,
    plot_height: int,
    frame_boundaries: list[int] | None = None,
    y_padding: float = 0.5,
    y_tick_increment: float | None = None,
    y_axis_label: str | None = None,
) -> str:
    """Return SVG HTML for one frame-indexed trace plot."""
    plot_width = 360
    margin_left = 48
    margin_right = 18
    margin_top = 18
    margin_bottom = 34
    inner_width = plot_width - margin_left - margin_right
    inner_height = plot_height - margin_top - margin_bottom
    colors = [
        "#2b6cb0",
        "#c2410c",
        "#15803d",
        "#7c3aed",
        "#be123c",
        "#0f766e",
        "#854d0e",
        "#4338ca",
    ]

    y_values = [y for item in series for y in item["y"]]
    observed_y_min = min(y_values)
    observed_y_max = max(y_values)
    y_min = float(observed_y_min) - y_padding
    y_max = float(observed_y_max) + y_padding
    y_ticks = (
        _integer_y_ticks(y_min, y_max)
        if y_tick_increment is None
        else _fixed_y_ticks(y_min, y_max, y_tick_increment)
    )

    x_denom = max(n_frames - 1, 1)

    def x_px(frame_index: int) -> float:
        return margin_left + inner_width * (frame_index / x_denom)

    def y_px(energy: float) -> float:
        return margin_top + inner_height * ((y_max - energy) / (y_max - y_min))

    boundary_parts = []
    for boundary in frame_boundaries or []:
        if boundary <= 0 or boundary >= n_frames:
            continue
        x = x_px(boundary)
        boundary_parts.append(
            f'<line class="molzen_cat_boundary_UNIQUEID" x1="{x:.2f}" '
            f'y1="{margin_top}" x2="{x:.2f}" '
            f'y2="{margin_top + inner_height}" stroke="#9ca3af" '
            'stroke-width="1" stroke-dasharray="4 4" opacity="0.55" />'
        )

    path_parts = []
    label_parts = []
    for i, item in enumerate(series):
        color_index = item.get("color_index", i)
        if not isinstance(color_index, int):
            color_index = i
        color = colors[color_index % len(colors)]
        points = [
            (x_px(int(frame_index)), y_px(float(energy)))
            for frame_index, energy in zip(item["x"], item["y"])
        ]
        if not points:
            continue
        if len(points) == 1:
            x, y = points[0]
            path_parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5" fill="{color}" />'
            )
        else:
            path = " ".join(
                f"{'M' if j == 0 else 'L'} {x:.2f} {y:.2f}"
                for j, (x, y) in enumerate(points)
            )
            path_parts.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.5" />'
            )
        label_parts.append(
            f'<span style="color: {color}; white-space: nowrap;">'
            f"{html.escape(str(item['label']))}</span>"
        )

    tick_parts = []
    for tick in y_ticks:
        y = y_px(float(tick))
        tick_label = html.escape(_tick_label(tick))
        tick_parts.append(
            f'<line x1="{margin_left - 4}" y1="{y:.2f}" '
            f'x2="{margin_left + inner_width}" y2="{y:.2f}" '
            'stroke="#e5e7eb" stroke-width="1" />'
            f'<text x="{margin_left - 8}" y="{y + 4:.2f}" fill="#555" '
            f'text-anchor="end">{tick_label}</text>'
        )

    x_axis_y = margin_top + inner_height
    x_tick_parts = []
    x_ticks = [
        (margin_left, "0", "middle"),
        (margin_left + inner_width / 2, str(n_frames // 2), "middle"),
        (margin_left + inner_width, str(n_frames), "middle"),
    ]
    for x, tick_label, text_anchor in x_ticks:
        x_tick_parts.append(
            f'<line x1="{x:.2f}" y1="{x_axis_y}" x2="{x:.2f}" '
            f'y2="{x_axis_y + 4}" stroke="#444" stroke-width="1" />'
            f'<text x="{x:.2f}" y="{x_axis_y + 16}" fill="#555" '
            f'text-anchor="{text_anchor}">{html.escape(tick_label)}</text>'
        )

    current_frame = 0 if frame is None else frame
    cursor_x = x_px(current_frame)
    escaped_title = html.escape(title)
    y_label_part = ""
    if y_axis_label:
        y_label_part = (
            f'<text transform="translate(18.00 '
            f'{margin_top + inner_height / 2:.2f}) rotate(-90)" '
            'fill="#555" text-anchor="middle" dominant-baseline="middle" '
            f'font-size="12">{html.escape(y_axis_label)}</text>'
        )
    return f"""
<div id="molzen_{panel_id}_panel_UNIQUEID" style="width: {plot_width}px; font: 12px {_PLOT_FONT_FAMILY};">
  <svg id="molzen_{panel_id}_svg_UNIQUEID" width="{plot_width}" height="{plot_height}" viewBox="0 0 {plot_width} {plot_height}" role="img" aria-label="{escaped_title}" style="font-family: {_PLOT_FONT_FAMILY}; font-size: 12px;">
    <rect x="0" y="0" width="{plot_width}" height="{plot_height}" fill="white" />
    {y_label_part}
    {"".join(tick_parts)}
    <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + inner_height}" stroke="#444" stroke-width="1" />
    <line x1="{margin_left}" y1="{margin_top + inner_height}" x2="{margin_left + inner_width}" y2="{margin_top + inner_height}" stroke="#444" stroke-width="1" />
    {"".join(x_tick_parts)}
    <text x="{plot_width / 2:.2f}" y="14" fill="#222" text-anchor="middle" font-size="14" font-weight="600">{escaped_title}</text>
    <text x="{margin_left + inner_width / 2:.2f}" y="{plot_height - 5}" fill="#555" text-anchor="middle">Frame</text>
    {"".join(boundary_parts)}
    {"".join(path_parts)}
    <line class="molzen_plot_cursor_UNIQUEID" x1="{cursor_x:.2f}" y1="{margin_top}" x2="{cursor_x:.2f}" y2="{margin_top + inner_height}" stroke="#111" stroke-width="1" opacity="0.8" />
  </svg>
  <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-left: {margin_left}px; width: {inner_width}px; font-size: 14px; line-height: 1.3;">{"".join(label_parts)}</div>
</div>
"""


def _add_state_property_plots(
    view: Any,
    *,
    records: list[dict[str, Any]] | None,
    metadata: dict[str, Any] | None,
    n_frames: int,
    frame: int | None,
    height: str | int,
) -> None:
    """Attach inline SVG state-property plots next to the molecule viewer."""
    energy_series = _excited_state_energy_series(records)
    oscillator_series = _excited_state_oscillator_strength_series(records)
    if not energy_series and not oscillator_series:
        return

    frame_boundaries = _cat_frame_boundaries(metadata, n_frames)
    plot_height = max(_pixel_size(height, 300), 180)
    plot_html = """
<div id="molzen_show_outer_UNIQUEID" style="display: flex; justify-content: center; width: 100%;">
<div id="molzen_show_row_UNIQUEID" style="display: inline-flex; align-items: flex-start; gap: 12px; flex-wrap: nowrap;">
"""
    graph_html = ""
    if energy_series:
        graph_html += _state_trace_plot_html(
            "energy",
            "Adiabatic state energies (eV)",
            energy_series,
            n_frames=n_frames,
            frame=frame,
            plot_height=plot_height,
            frame_boundaries=frame_boundaries,
            y_axis_label="Energy (eV)",
        )
    if oscillator_series:
        graph_html += _state_trace_plot_html(
            "oscillator",
            "Oscillator strength",
            oscillator_series,
            n_frames=n_frames,
            frame=frame,
            plot_height=plot_height,
            frame_boundaries=frame_boundaries,
            y_padding=0.2,
            y_tick_increment=0.25,
        )

    x_margin_left = 48
    x_inner_width = 360 - 48 - 18
    script = f"""
window.molzenPlotFrameToX_UNIQUEID = function(frame) {{
    var nFrames = {n_frames};
    var xDenom = Math.max(nFrames - 1, 1);
    return {x_margin_left} + {x_inner_width} * (frame / xDenom);
}};
window.molzenUpdateEnergyFrame_UNIQUEID = function(frame) {{
    var cursors = document.querySelectorAll(".molzen_plot_cursor_UNIQUEID");
    for(var i = 0; i < cursors.length; i++) {{
        var cursor = cursors[i];
        var x = window.molzenPlotFrameToX_UNIQUEID(frame);
        cursor.setAttribute("x1", String(x));
        cursor.setAttribute("x2", String(x));
    }}
}};
"""
    view.startjs = view.startjs.replace(
        '<div id="3dmolviewer_UNIQUEID"',
        plot_html + '<div id="3dmolviewer_UNIQUEID"',
        1,
    )
    view.startjs = view.startjs.replace(
        "margin-left: auto; margin-right: auto; width:",
        "width:",
        1,
    )
    view.startjs = view.startjs.replace(
        "</div>\n<script>\n",
        f"</div>\n{graph_html}</div></div>\n<script>\n",
        1,
    )
    view.startjs += script


def _cat_frame_boundaries(metadata: dict[str, Any] | None, n_frames: int) -> list[int]:
    """Return valid frame-boundary indices from concatenation metadata."""
    if not metadata:
        return []

    cat_metadata = metadata.get("cat_frames")
    if not isinstance(cat_metadata, dict):
        return []

    raw_boundaries = cat_metadata.get("frame_boundaries")
    if raw_boundaries is None:
        segments = cat_metadata.get("segments")
        if not isinstance(segments, list):
            return []
        raw_boundaries = [
            segment.get("frame_start")
            for segment in segments[1:]
            if isinstance(segment, dict)
        ]

    boundaries: list[int] = []
    for value in raw_boundaries:
        try:
            boundary = int(value)
        except (TypeError, ValueError):
            continue
        if 0 < boundary < n_frames and boundary not in boundaries:
            boundaries.append(boundary)
    return sorted(boundaries)


def _apply_py3dmol_style(view: Any, style: dict[str, Any] | None) -> None:
    """Apply either a caller-provided style or Molzen's default molecule style."""
    if style is not None:
        view.setStyle(style)
        return

    view.setStyle({}, {"stick": {"radius": 0.12}, "sphere": {"scale": 0.25}})
    view.setStyle({"elem": "H"}, {"stick": {"radius": 0.06}, "sphere": {"scale": 0.21}})


def _atom_hover_label_name(row: np.void, fallback_index: int) -> str:
    """Return the atom name to show in a hover label."""
    atom_name = str(row["atom_name"]).strip()
    if atom_name:
        return atom_name

    atom_type = str(row["atom_type"]).strip()
    if atom_type:
        return atom_type

    element = str(row["element"]).strip().capitalize()
    if element:
        return element

    return f"Atom {fallback_index + 1}"


def _atom_hover_label_records(atom_records: np.ndarray) -> list[dict[str, int | str]]:
    """Return compact per-atom label data for browser-side hover callbacks."""
    labels: list[dict[str, int | str]] = []
    for row_position, row in enumerate(atom_records):
        try:
            atom_index = int(row["atom_index"])
        except (TypeError, ValueError):
            atom_index = row_position
        labels.append(
            {
                "name": _atom_hover_label_name(row, row_position),
                "index": atom_index,
            }
        )
    return labels


def _add_py3dmol_atom_hover_labels(view: Any, atom_records: np.ndarray) -> None:
    """Attach atom-name and molecule-index hover labels to a py3Dmol view."""
    labels_json = json.dumps(_atom_hover_label_records(atom_records))
    script = f"""
var molzenAtomHoverLabels_UNIQUEID = {labels_json};
function molzenAtomHoverNumber_UNIQUEID(value) {{
    if(typeof value === "number" && Number.isFinite(value)) {{
        return Math.trunc(value);
    }}
    if(typeof value === "string" && value.trim() !== "") {{
        var parsed = parseInt(value, 10);
        if(Number.isFinite(parsed)) {{
            return parsed;
        }}
    }}
    return null;
}}
function molzenAtomHoverLabelIndex_UNIQUEID(atom) {{
    if(!atom) {{
        return -1;
    }}
    var index = molzenAtomHoverNumber_UNIQUEID(atom.index);
    if(index !== null && index >= 0 && index < molzenAtomHoverLabels_UNIQUEID.length) {{
        return index;
    }}
    var serial = molzenAtomHoverNumber_UNIQUEID(atom.serial);
    if(serial !== null) {{
        var oneBasedIndex = serial - 1;
        if(oneBasedIndex >= 0 && oneBasedIndex < molzenAtomHoverLabels_UNIQUEID.length) {{
            return oneBasedIndex;
        }}
        if(serial >= 0 && serial < molzenAtomHoverLabels_UNIQUEID.length) {{
            return serial;
        }}
    }}
    return -1;
}}
function molzenRemoveAtomHoverLabel_UNIQUEID(viewer) {{
    if(window.molzenAtomHoverActiveLabel_UNIQUEID) {{
        viewer.removeLabel(window.molzenAtomHoverActiveLabel_UNIQUEID);
        window.molzenAtomHoverActiveLabel_UNIQUEID = null;
    }}
}}
window.molzenClearAtomHoverLabel_UNIQUEID = function() {{
    molzenRemoveAtomHoverLabel_UNIQUEID(viewer_UNIQUEID);
}};
function molzenAtomHoverCallback_UNIQUEID(atom, viewer) {{
        if(!atom) {{
            return;
        }}
        var labelIndex = molzenAtomHoverLabelIndex_UNIQUEID(atom);
        if(labelIndex < 0) {{
            return;
        }}
        var atomInfo = molzenAtomHoverLabels_UNIQUEID[labelIndex];
        var labelText = atomInfo.name + " (index " + atomInfo.index + ")";
        molzenRemoveAtomHoverLabel_UNIQUEID(viewer);
        window.molzenAtomHoverActiveLabel_UNIQUEID = viewer.addLabel(labelText, {{
            position: atom,
            backgroundColor: "#111827",
            backgroundOpacity: 0.85,
            borderThickness: 0,
            fontColor: "#ffffff",
            fontSize: 12,
            inFront: true,
            padding: 4,
            screenOffset: {{x: 16, y: -16}}
        }});
        viewer.render();
}}
function molzenAtomUnhoverCallback_UNIQUEID(atom, viewer) {{
    molzenRemoveAtomHoverLabel_UNIQUEID(viewer);
    viewer.render();
}}
window.molzenEnableAtomHoverLabels_UNIQUEID = function() {{
    viewer_UNIQUEID.setHoverable(
        {{}},
        true,
        molzenAtomHoverCallback_UNIQUEID,
        molzenAtomUnhoverCallback_UNIQUEID
    );
}};
window.molzenEnableAtomHoverLabels_UNIQUEID();
"""
    view.startjs += script


def show_molecule_py3dmol(
    mol: Molecule,
    *,
    width: str | int = "500px",
    height: str | int = "300px",
    frame: int | None = None,
    style: dict[str, Any] | None = None,
    animate: bool = False,
    show_slider: bool = True,
    export_controls: bool = False,
    gif_delay_ms: int = 120,
    gif_total_time: float | None = None,
    gif_bounce: bool = False,
    png_scale: float = 2.0,
    atom_hover_labels: bool = True,
    atom_hover_duration: float = 0.25,
) -> Any:
    """Return a py3Dmol view for a molecule.

    Args:
        mol: The molecule to visualize.
        width: The width of the visualization (e.g., "500px").
        height: The height of the visualization (e.g., "300px").
        frame: Optional frame index to show. When omitted, multi-frame molecules
            are loaded as an interactive trajectory.
        style: Optional 3Dmol.js style dictionary. Defaults to stick style.
        animate: Whether to start playback for multi-frame molecules.
        show_slider: Whether to add a frame slider for multi-frame molecules.
        export_controls: Whether to add browser-side GIF export controls for
            multi-frame molecules.
        gif_delay_ms: Delay between exported GIF frames in milliseconds.
        gif_total_time: Total exported GIF duration in seconds. When provided,
            this overrides gif_delay_ms.
        gif_bounce: Whether exported GIF frames should play forward and then
            backward to the first frame.
        png_scale: Pixel scale for PNG and clipboard exports. A value of 2.0
            exports twice the notebook display dimensions.
        atom_hover_labels: Whether atoms should show their name and molecule
            index when hovered.
        atom_hover_duration: Delay in seconds before atom hover labels appear.
    """
    py3dmol = _require_py3dmol()
    atom_records = mol.atom_records
    if atom_records is None or len(atom_records) == 0:
        raise ValueError("Cannot visualize an empty molecule.")

    view = py3dmol.view(width=width, height=height)
    _center_py3dmol_view(view)
    if atom_hover_labels:
        _set_py3dmol_hover_duration(view, atom_hover_duration)

    if frame is None and _frame_count(atom_records) > 1:
        n_frames = _frame_count(atom_records)
        view.addModelsAsFrames(_xyz_text(atom_records), "xyz")
        _apply_py3dmol_style(view, style)
        if atom_hover_labels:
            _add_py3dmol_atom_hover_labels(view, atom_records)
        _add_state_property_plots(
            view,
            records=mol.excited_state_records,
            metadata=mol.metadata,
            n_frames=n_frames,
            frame=frame,
            height=height,
        )
        if show_slider:
            _add_py3dmol_frame_slider(view, n_frames, width)
        if export_controls:
            _add_py3dmol_gif_export_controls(
                view,
                n_frames,
                width,
                delay_ms=gif_delay_ms,
                bounce=gif_bounce,
                total_time=gif_total_time,
                png_scale=png_scale,
            )
        if animate:
            view.animate({"loop": "forward"})
    else:
        frame_index = 0 if frame is None else _coerce_frame_index(atom_records, frame)
        view.addModel(_xyz_text(atom_records, frame=frame_index), "xyz")
        _apply_py3dmol_style(view, style)
        if atom_hover_labels:
            _add_py3dmol_atom_hover_labels(view, atom_records)
        _add_state_property_plots(
            view,
            records=mol.excited_state_records,
            metadata=mol.metadata,
            n_frames=_frame_count(atom_records),
            frame=frame_index,
            height=height,
        )

    view.zoomTo()
    return view


show_molecule_py3Dmol = show_molecule_py3dmol
show_molecule = show_molecule_py3dmol
