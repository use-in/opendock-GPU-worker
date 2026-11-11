
"""Validate the backend -> GPU Worker JSON contract."""

from __future__ import annotations

from dataclasses import dataclass
import re


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

JOB_ID = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z"
)


@dataclass(frozen=True)
class Options:
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1


@dataclass(frozen=True)
class Job:
    job_id: str
    attempt: int
    protein_sequence: str
    ligand_smiles: str
    msa_mode: str
    msa_s3_uri: str | None
    output_s3_prefix: str | None
    options: Options


def _number(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(
            f"{name}: integer {low}..{high} required"
        )
    return value


def parse_job(
    payload: dict,
    *,
    demo: bool = False,
) -> Job:

    if not isinstance(payload, dict):
        raise ValueError("Job must be an object")

    allowed = {
        "job_id",
        "attempt",
        "protein_sequence",
        "ligand_smiles",
        "msa",
        "output_s3_prefix",
        "options",
    }

    if set(payload) - allowed:
        raise ValueError("Unknown job fields")

    jid = payload.get("job_id")

    if (
        not isinstance(jid, str)
        or JOB_ID.fullmatch(jid) is None
    ):
        raise ValueError("Invalid job_id")

    attempt = _number(
        payload.get("attempt", 1),
        "attempt",
        1,
        20,
    )

    seq = payload.get("protein_sequence")

    if not isinstance(seq, str):
        raise ValueError(
            "protein_sequence must be a string"
        )

    seq = seq.strip().upper()

    if (
        not 20 <= len(seq) <= 5000
        or not set(seq) <= AMINO_ACIDS
    ):
        raise ValueError(
            "Protein requires 20..5000 "
            "canonical amino acids"
        )

    smiles = payload.get("ligand_smiles")

    if (
        not isinstance(smiles, str)
        or not 1 <= len(smiles) <= 1024
    ):
        raise ValueError(
            "Invalid ligand_smiles"
        )

    if (
        smiles != smiles.strip()
        or any(ord(c) < 32 for c in smiles)
    ):
        raise ValueError(
            "Invalid SMILES characters"
        )

    msa = payload.get("msa")

    if (
        not isinstance(msa, dict)
        or set(msa) - {"mode", "s3_uri"}
    ):
        raise ValueError(
            "msa must contain mode "
            "and optional s3_uri"
        )

    mode = msa.get("mode")
    uri = msa.get("s3_uri")

    if mode == "s3":
        if (
            not isinstance(uri, str)
            or not uri.endswith(".a3m")
        ):
            raise ValueError(
                "msa.s3_uri must be an "
                ".a3m S3 object"
            )

    elif mode == "empty" and demo and uri is None:
        pass

    else:
        raise ValueError(
            "Production jobs require msa.mode=s3"
        )

    output = payload.get("output_s3_prefix")

    if demo:
        if output is not None:
            raise ValueError(
                "Demo must not use S3 output"
            )

    else:
        from worker.storage import parse_s3_uri

        _, key = parse_s3_uri(
            output,
            is_prefix=True,
        )

        expected = (
            f"jobs/{jid}/attempt-{attempt}/"
        )

        if key != expected:
            raise ValueError(
                "Unexpected output_s3_prefix key"
            )

    raw = payload.get("options", {})

    if (
        not isinstance(raw, dict)
        or set(raw) - {
            "recycling_steps",
            "sampling_steps",
            "diffusion_samples",
            "affinity",
        }
    ):
        raise ValueError("Unknown options")

    if raw.get("affinity", False) is not False:
        raise ValueError(
            "Affinity needs a separate model; "
            "unsupported here"
        )

    opts = Options(
        _number(
            raw.get("recycling_steps", 3),
            "recycling_steps",
            1,
            10,
        ),
        _number(
            raw.get("sampling_steps", 200),
            "sampling_steps",
            1,
            200,
        ),
        _number(
            raw.get("diffusion_samples", 1),
            "diffusion_samples",
            1,
            5,
        ),
    )

    return Job(
        jid,
        attempt,
        seq,
        smiles,
        mode,
        uri,
        output,
        opts,
    )
