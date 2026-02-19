from pathlib import Path
import pytest

from molzen.io.mos import generate_nto_cubes

pytestmark = pytest.mark.local_only


def test_cube_generation():
    """Test .cub generation from .molden files."""
    this_file = Path(__file__)
    molden_file = this_file.parent / "data" / "nto_S0-S1.molden"
    molden_file_str = str(molden_file.resolve())

    multiwfn_path = str(
        Path("~/software/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn_noGUI")
        .expanduser()
        .resolve()
    )

    generate_nto_cubes(molden_file_str, occ_thresh=0.1, multiwfn_path=multiwfn_path)
    cubfiles = list(Path(molden_file).parent.glob("*.cub"))
    assert len(cubfiles) == 4
    # delete the generated .cub files to keep the test directory clean.
    for cubfile in cubfiles:
        cubfile.unlink()
