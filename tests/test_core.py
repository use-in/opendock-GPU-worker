
import pytest

from worker.contracts import parse_job
from worker.routing import route_job
from worker.preprocessing import make_yaml


PROTEIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLED"
    "GRTLSDYNIQKESTLHLVLRLRGG"
)


def payload():
    return {
        "job_id": "test",
        "protein_sequence": PROTEIN,
        "ligand_smiles": "CCO",
        "msa": {
            "mode": "empty"
        },
        "options": {
            "sampling_steps": 20
        },
    }


def test_empty_restricted():

    with pytest.raises(ValueError):
        parse_job(payload())


def test_smoke_route_small():

    job = parse_job(
        payload(),
        demo=True,
    )

    assert route_job(job)["pool"] == "small"
    assert route_job(job)["gpu"] == "H100"


def test_invalid_smiles():

    job = parse_job(
        {
            **payload(),
            "ligand_smiles": "not-a-smiles",
        },
        demo=True,
    )

    with pytest.raises(ValueError):
        route_job(job)


def test_yaml(tmp_path):

    import yaml

    job = parse_job(
        payload(),
        demo=True,
    )

    path = make_yaml(
        job,
        tmp_path / "complex.yaml",
        None,
    )

    obj = yaml.safe_load(
        path.read_text()
    )

    assert (
        obj["sequences"][0]["protein"]["msa"]
        == "empty"
    )

    assert (
        obj["sequences"][1]["ligand"]["smiles"]
        == "CCO"
    )


def test_reject_affinity():

    data = payload()

    data["options"]["affinity"] = True

    with pytest.raises(ValueError):
        parse_job(
            data,
            demo=True,
        )
