
"""Boltz YAML generation and MSA validation."""

from __future__ import annotations

from pathlib import Path
import yaml

from worker.contracts import Job


def make_yaml(
    job: Job,
    path: Path,
    msa_path: Path | None,
) -> Path:

    if job.msa_mode == "s3" and msa_path is None:
        raise ValueError(
            "MSA was not downloaded"
        )

    payload = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": job.protein_sequence,
                    "msa": (
                        str(msa_path)
                        if msa_path
                        else "empty"
                    ),
                }
            },
            {
                "ligand": {
                    "id": "B",
                    "smiles": job.ligand_smiles,
                }
            },
        ],
    }

    path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return path


def validate_msa(
    path: Path,
    query_sequence: str,
) -> None:

    if (
        not path.exists()
        or not 0 < path.stat().st_size
        <= 100 * 1024 * 1024
    ):
        raise ValueError(
            "Missing/invalid MSA"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        if not f.readline().startswith(">"):
            raise ValueError(
                "A3M missing FASTA header"
            )

        fragments = []

        for line in f:
            if line.startswith(">"):
                break

            fragments.append(
                line.strip()
            )

    # A3M lowercase characters are insertions.
    # Remove lowercase insertions and alignment gaps.

    query = "".join(
        c
        for c in "".join(fragments)
        if c.isupper() and c not in "-."
    )

    if query != query_sequence:
        raise ValueError(
            "A3M query row differs "
            "from protein_sequence"
        )
