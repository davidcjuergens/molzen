"""PubChem loading tests; no network access required."""

import io
import json
from urllib.error import HTTPError, URLError

import numpy as np
import pytest

from molzen import io as mzio
from molzen.io import pubchem


@pytest.fixture
def record():
    return {
        "PC_Compounds": [
            {
                "atoms": {"aid": [1, 2, 3], "element": [8, 1, 1]},
                "coords": [
                    {
                        "aid": [3, 1, 2],
                        "conformers": [
                            {
                                "x": [-0.7, 0.0, 0.7],
                                "y": [0.5, 0.0, 0.5],
                                "z": [0.0, 0.0, 0.0],
                            }
                        ],
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize("cid", [962, "962", np.int64(962)])
def test_from_pubchem(monkeypatch, record, tmp_path, cid):
    def open_response(url, *, timeout):
        assert url.endswith("/cid/962/JSON?record_type=3d")
        assert timeout == 5
        return io.BytesIO(json.dumps(record).encode())

    monkeypatch.setattr(pubchem, "urlopen", open_response)
    mol = mzio.Molecule.from_pubchem(cid, timeout=5)
    assert mol.xyz.shape == (3, 3)
    np.testing.assert_allclose(mol.xyz, [[0, 0, 0], [0.7, 0.5, 0], [-0.7, 0.5, 0]])
    assert mol.elements == ["O", "H", "H"]
    np.testing.assert_array_equal(mol.Z, [8, 1, 1])
    assert mol.metadata["pubchem_cid"] == 962
    path = tmp_path / "water.xyz"
    mol.to_xyz(str(path))
    np.testing.assert_allclose(mzio.Molecule.from_xyz(str(path)).xyz, mol.xyz)


@pytest.mark.parametrize("cid", [True, 0, -1, 1.5, None, "", "1,2", "1.0", "abc"])
def test_invalid_cid_does_not_request(monkeypatch, cid):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid CID must be rejected before requesting PubChem")

    monkeypatch.setattr(pubchem, "urlopen", unexpected)
    with pytest.raises(ValueError, match="cid must"):
        mzio.Molecule.from_pubchem(cid)


@pytest.mark.parametrize(
    "error, exception, message",
    [
        (HTTPError("url", 404, "Not Found", {}, None), ValueError, "no 3D conformer"),
        (HTTPError("url", 503, "Unavailable", {}, None), OSError, "HTTP 503"),
        (URLError("offline"), OSError, "offline"),
        (TimeoutError("timed out"), OSError, "timed out"),
    ],
)
def test_request_errors(monkeypatch, error, exception, message):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(pubchem, "urlopen", fail)
    with pytest.raises(exception, match=message):
        mzio.Molecule.from_pubchem(962)


@pytest.mark.parametrize(
    "problem", ["2d", "mismatched_ids", "nan", "empty", "invalid_json"]
)
def test_bad_response(monkeypatch, record, problem):
    coords = record["PC_Compounds"][0]["coords"][0]
    if problem == "2d":
        del coords["conformers"][0]["z"]
    elif problem == "mismatched_ids":
        coords["aid"] = [1, 2, 4]
    elif problem == "nan":
        coords["conformers"][0]["x"][0] = float("nan")
    elif problem == "empty":
        record["PC_Compounds"] = []
    raw = b"not json" if problem == "invalid_json" else json.dumps(record).encode()
    monkeypatch.setattr(pubchem, "urlopen", lambda *a, **kw: io.BytesIO(raw))
    with pytest.raises(ValueError, match="Invalid PubChem 3D structure"):
        mzio.Molecule.from_pubchem(962)
