
"""Single-job H100 pool routing.

Thresholds are provisional, not VRAM guarantees.
"""

from __future__ import annotations

from math import ceil
from worker.contracts import Job


SMALL_MAX = 1024
MEDIUM_MAX = 2560
LARGE_MAX = 5120

HEADROOM = 1.20


def route_job(job: Job) -> dict:

    from rdkit import Chem

    molecule = Chem.MolFromSmiles(
        job.ligand_smiles
    )

    if molecule is None:
        raise ValueError(
            "Invalid ligand SMILES"
        )

    atoms = molecule.GetNumHeavyAtoms()

    if atoms < 1:
        raise ValueError(
            "Ligand requires a heavy atom"
        )

    estimate = ceil(
        (
            len(job.protein_sequence)
            + atoms
        ) * HEADROOM
    )

    if estimate <= SMALL_MAX:
        pool = "small"

    elif estimate <= MEDIUM_MAX:
        pool = "medium"

    elif estimate <= LARGE_MAX:
        pool = "large"

    else:
        raise ValueError(
            "Input exceeds unvalidated routing limit"
        )

    return {
        "pool": pool,
        "gpu": "H100",
        "protein_residues": len(
            job.protein_sequence
        ),
        "ligand_heavy_atoms": atoms,
        "routing_size": estimate,
        "routing_is_estimate": True,
    }
