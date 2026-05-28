"""Helpers for visualizing Molzen objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from molzen.io.molecule import Molecule


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
        var framePromise = viewer_UNIQUEID.setFrame(frame);
        if(framePromise && typeof framePromise.then === "function") {
            framePromise.then(function() { viewer_UNIQUEID.render(); });
        } else {
            viewer_UNIQUEID.render();
        }
    });
}
"""
    view.startjs = view.startjs.replace(
        "</div>\n<script>\n", f"</div>\n{slider_html}<script>\n", 1
    )
    view.startjs += script


def _apply_py3dmol_style(view: Any, style: dict[str, Any] | None) -> None:
    """Apply either a caller-provided style or Molzen's default molecule style."""
    if style is not None:
        view.setStyle(style)
        return

    view.setStyle({}, {"stick": {"radius": 0.12}, "sphere": {"scale": 0.25}})
    view.setStyle({"elem": "H"}, {"stick": {"radius": 0.06}, "sphere": {"scale": 0.16}})


def show_molecule_py3dmol(
    mol: Molecule,
    *,
    width: str | int = "300px",
    height: str | int = "300px",
    frame: int | None = None,
    style: dict[str, Any] | None = None,
    animate: bool = False,
    show_slider: bool = True,
) -> Any:
    """Return a py3Dmol view for a molecule.

    Args:
        mol: The molecule to visualize.
        width: The width of the visualization (e.g., "300px").
        height: The height of the visualization (e.g., "300px").
        frame: Optional frame index to show. When omitted, multi-frame molecules
            are loaded as an interactive trajectory.
        style: Optional 3Dmol.js style dictionary. Defaults to stick style.
        animate: Whether to start playback for multi-frame molecules.
        show_slider: Whether to add a frame slider for multi-frame molecules.
    """
    py3dmol = _require_py3dmol()
    atom_records = mol.atom_records
    if atom_records is None or len(atom_records) == 0:
        raise ValueError("Cannot visualize an empty molecule.")

    view = py3dmol.view(width=width, height=height)
    _center_py3dmol_view(view)

    if frame is None and _frame_count(atom_records) > 1:
        n_frames = _frame_count(atom_records)
        view.addModelsAsFrames(_xyz_text(atom_records), "xyz")
        _apply_py3dmol_style(view, style)
        if show_slider:
            _add_py3dmol_frame_slider(view, n_frames, width)
        if animate:
            view.animate({"loop": "forward"})
    else:
        view.addModel(_xyz_text(atom_records, frame=frame), "xyz")
        _apply_py3dmol_style(view, style)

    view.zoomTo()
    return view


show_molecule_py3Dmol = show_molecule_py3dmol
show_molecule = show_molecule_py3dmol
