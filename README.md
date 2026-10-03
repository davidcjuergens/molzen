<p align="center">
  <img src="molzen/img/molzen4.png" alt="molzen" width="700" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/style-ruff-dafe6d" />
</p>

# molzen

Python utilities for computational chemistry: read and write molecular structures,
work with trajectories, parse ORCA and TeraChem outputs, and visualize molecules.

## Setup

Requires Python 3.11+ and `uv`. CI uses Python 3.13.

```bash
git clone git@github.com:davidcjuergens/molzen.git
cd molzen
uv sync --frozen
```

Run Python scripts with `uv run python script.py`, or start an interactive session
with `uv run python`. These commands use the project's environment and dependencies.

## Quick start

Create a molecule and save it as XYZ:

```python
import numpy as np
from molzen.io import Molecule

water = Molecule(
    elements=["O", "H", "H"],
    xyz=np.array([
        [0.000, 0.000, 0.000],
        [0.758, 0.000, 0.504],
        [-0.758, 0.000, 0.504],
    ]),
)
water.to_xyz("water.xyz")

mol = Molecule.from_xyz("water.xyz")
print(mol.elements)   # ["O", "H", "H"]
print(mol.xyz.shape)  # (3, 3)
```

`mol.xyz` has shape `(n_atoms, 3)` for a single frame and
`(n_frames, n_atoms, 3)` for multiple frames.

Load other structures or calculation outputs with the corresponding reader:

```python
protein = Molecule.from_pdb("protein.pdb")
orca = Molecule.from_orca_stdout("orca.out")
terachem = Molecule.from_terachem_stdout("terachem.out")
```

TeraChem output loading may also require the coordinate or trajectory files
referenced by the calculation, so keep its input and scratch files accessible.

In a Jupyter notebook using the project environment, display an interactive
py3Dmol view with:

```python
mol.show()
```

## Interactive editing

Install the optional notebook editor into the environment used by your kernel:

```bash
uv sync --extra edit --inexact
```

The `--inexact` option keeps separately installed notebook packages. Alternatively,
use `pip install -e '.[edit]'`. Restart an existing kernel after installation.

```python
from molzen.io import Molecule

mol = Molecule.from_pubchem(1174)  # uracil; downloading requires internet
editor = mol.show(edit=True, width=760, height=420)
display(editor)
```

Click an atom, or select it in the atom list. Try changing an oxygen to sulfur and
then **Undo**. Select a hydrogen and choose **Replace H with methyl**; the torsion
field controls its initial orientation. This creates idealized starting geometry,
without energy minimization or automatic hydrogen/charge adjustment. Select two
atoms to set or remove a bond. Undo and redo restore both atoms and connectivity.

```python
edited = editor.molecule       # independent copy of the latest accepted edit
edited.to_pdb("edited.pdb")    # atoms and connectivity; PDB does not retain orders
edited.to_mol2("edited.mol2")  # supported bond orders
edited.to_npy("edited.npy")   # complete molzen records, graph, and provenance
```

Editing leaves `mol` untouched. A trajectory requires an explicit `frame=...` and
produces a single-frame copy. Calculated properties and spin multiplicity are
cleared after an edit; inspect charge/spin and optimize the geometry as appropriate
before a new calculation. Original metadata is retained as source provenance.

The editor bundles 3Dmol.js 2.5.4 and works without external renderer downloads
once its notebook widget support is installed. Standard `mol.show()` continues to
use py3Dmol. See [the example notebook](examples/edit_molecule.ipynb) for a walkthrough.

### Connectivity

`mol.bonds` is either `None` (not supplied) or a `BondGraph`, whose `.records` table
contains `a`, `b`, `order`, and `source`. Endpoints refer to stable values in
`atom_records['atom_index']`, not array positions or PDB serial numbers. Graph
coverage is `partial` or `complete`; unknown bond order is stored explicitly.
PubChem and MOL2 connectivity is imported, and PDB `CONECT` is imported as partial
connectivity with unknown orders. NPY/HDF5 retain the full graph and atom IDs.
MOL2 preserves connectivity and supported bond types, but does not perform atom
typing or infer partial charges. Multi-model PDB editing is not supported yet.

For coordinate-only inputs, **Suggest bonds** previews distance-based candidates
in gold. **Accept suggestions** adds them with unknown orders. Use **Set bond** to
confirm an inferred hydrogen-parent bond as single before methyl replacement.
Inference uses covalent radii for common nonmetal elements, skips incompatible
alternate locations, and reports suspicious contacts/coordination. It does not
assign aromaticity, multiple bonds, or metal coordination.

The same operations are available without notebook dependencies:

```python
from molzen.editing import EditSession

session = EditSession(mol)
# These IDs refer to mol.atom_records['atom_index'].
hydrogen_id = int(mol.atom_records["atom_index"][mol.atom_records["element"] == "H"][0])
session.apply("replace_hydrogen", atom_id=hydrogen_id, torsion=30)
edited = session.molecule
session.undo()
```
