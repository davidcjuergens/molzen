"""Optional anywidget interface for interactive molecule editing."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import anywidget
    import traitlets
except ImportError as exc:
    raise ImportError(
        "Interactive editing requires 'pip install molzen[edit]'."
    ) from exc

from molzen.editing import EditSession, close_contacts
from molzen.ptable import ALL_SYMBOLS

if TYPE_CHECKING:
    from molzen.io.molecule import Molecule

_STATIC = Path(__file__).parent / "static"
# Wrap the unmodified UMD distribution in an ESM-local CommonJS scope. This
# avoids CDN requests and does not interfere with a notebook's py3Dmol version.
_RENDERER = (
    "const molzen3Dmol = (() => { const module = {exports: {}}; const exports = module.exports;\n"
    "(function() {\n"
    + (_STATIC / "3Dmol-min.js").read_text()
    + "\n}).call(globalThis); return module.exports; })();\n"
)


class MoleculeEditor(anywidget.AnyWidget):
    """Render a copied molecule and exchange validated edits with the kernel.

    Args:
        molecule: Source molecule, which is never modified by this widget.
        frame: Frame to copy. Required for multi-frame molecules.
        width: Width in pixels or a CSS size string.
        height: Viewport height in pixels or a CSS size string.
    """

    _esm = _RENDERER + (_STATIC / "editor.js").read_text()
    _css = _STATIC / "editor.css"
    state = traitlets.Dict().tag(sync=True)
    width = traitlets.Unicode("650px").tag(sync=True)
    height = traitlets.Unicode("400px").tag(sync=True)

    def __init__(
        self,
        molecule: Molecule,
        *,
        frame: int | None = None,
        width: int | str = 650,
        height: int | str = 400,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._session = EditSession(molecule, frame=frame)
        self._seen: deque[str] = deque(maxlen=256)
        self.width = f"{width}px" if isinstance(width, int) else width
        self.height = f"{height}px" if isinstance(height, int) else height
        self.on_msg(self._handle_message)
        self._publish()

    @property
    def molecule(self) -> Molecule:
        """Return a copy of the latest accepted edit for saving or calculation."""
        return self._session.molecule

    def _handle_message(self, widget: Any, content: dict, buffers: list) -> None:
        """Reject stale or repeated commands before touching molecule state."""
        request = content.get("id")
        if not isinstance(request, str) or not request:
            return
        error = ""
        try:
            if request in self._seen:
                raise ValueError("This command has already been processed.")
            self._seen.append(request)
            if content.get("revision") != self._session.revision:
                raise ValueError("The molecule changed; please retry the edit.")
            action = content.get("action")
            args = content.get("args", {})
            if action == "undo":
                self._session.undo()
            elif action == "redo":
                self._session.redo()
            elif action == "suggest_bonds":
                self._session.suggest_bonds(**args)
            else:
                self._session.apply(action, **args)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            error = str(exc)
        self._publish(request=request, error=error)

    def _publish(self, *, request: str = "", error: str = "") -> None:
        """Send one consistent structure, history state, and acknowledgement."""
        mol = self._session.molecule
        atoms = []
        for row in mol.atom_records:
            x, y, z = map(float, row["coords"][0])
            atoms.append(
                {
                    "id": int(row["atom_index"]),
                    "elem": str(row["element"]),
                    "name": str(row["atom_name"]),
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
        bonds = (
            []
            if mol.bonds is None
            else [
                {
                    "a": int(r["a"]),
                    "b": int(r["b"]),
                    "order": str(r["order"]),
                    "source": str(r["source"]),
                }
                for r in mol.bonds.records
            ]
        )
        preview = self._session.suggestions
        self.state = {
            "atoms": atoms,
            "bonds": bonds,
            "elements": ALL_SYMBOLS,
            "coverage": "absent" if mol.bonds is None else mol.bonds.status,
            "revision": self._session.revision,
            "request": request,
            "error": error,
            "can_undo": self._session.can_undo,
            "can_redo": self._session.can_redo,
            "suggestions": []
            if preview is None
            else [{"a": int(r["a"]), "b": int(r["b"])} for r in preview.bonds.records],
            "diagnostics": close_contacts(mol)
            if preview is None
            else preview.diagnostics,
        }
