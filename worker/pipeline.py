
"""Single-job pipeline: MSA -> YAML -> inference -> S3."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from worker.contracts import parse_job

from worker.preprocessing import (
    make_yaml,
    validate_msa,
)

from worker.storage import (
    download_msa,
    upload_results,
)

from worker.boltz_resident import (
    predict_with_model,
)


def _manifest(
    out_dir: Path,
    job,
) -> dict:

    allowed = {
        ".cif",
        ".npz",
        ".json",
    }

    paths = sorted(
        p
        for p in out_dir.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower() in allowed
        )
    )

    if not any(
        p.suffix == ".cif"
        for p in paths
    ):
        raise RuntimeError(
            "No mmCIF structure generated"
        )

    if not any(
        p.name.startswith("confidence_")
        and p.suffix == ".json"
        for p in paths
    ):
        raise RuntimeError(
            "No confidence JSON generated"
        )

    files = []

    for path in paths:

        if (
            path.stat().st_size
            > 500 * 1024 * 1024
        ):
            raise RuntimeError(
                "Output artifact too large"
            )

        with path.open("rb") as f:
            checksum = hashlib.file_digest(
                f,
                "sha256",
            ).hexdigest()

        files.append({
            "path": path.relative_to(
                out_dir
            ).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": checksum,
        })

    return {
        "schema_version": 1,
        "job_id": job.job_id,
        "attempt": job.attempt,
        "status": "succeeded",
        "model": "boltz2",
        "files": files,
    }


def execute(
    payload: dict,
    *,
    model,
    cache_dir: Path,
    demo: bool = False,
) -> dict:

    job = parse_job(
        payload,
        demo=demo,
    )

    started = time.perf_counter()

    with TemporaryDirectory(
        prefix="opendock-job-"
    ) as tmp:

        root = Path(tmp)
        msa = None

        # ====================================
        # 1. Download MSA
        # ====================================

        if job.msa_mode == "s3":

            msa = root / "input.a3m"

            download_msa(
                job.msa_s3_uri,
                msa,
            )

            validate_msa(
                msa,
                job.protein_sequence,
            )

        # ====================================
        # 2. Validate ligand
        # ====================================

        from rdkit import Chem

        if Chem.MolFromSmiles(
            job.ligand_smiles
        ) is None:
            raise ValueError(
                "Invalid ligand SMILES"
            )

        # ====================================
        # 3. Generate Boltz YAML
        # ====================================

        yaml = make_yaml(
            job,
            root / "complex.yaml",
            msa,
        )

        # ====================================
        # 4. Resident inference
        # ====================================

        out = (
            root
            / "output"
            / "boltz_results_complex"
        )

        timings = predict_with_model(
            model,
            yaml,
            out,
            cache_dir,
            job.options,
        )

        # ====================================
        # 5. Collect outputs
        # ====================================

        result = _manifest(
            out,
            job,
        )

        result["timings_seconds"] = timings

        result[
            "processing_seconds_before_upload"
        ] = round(
            time.perf_counter() - started,
            3,
        )

        # ====================================
        # 6. Persist results
        # ====================================

        if demo:
            result["demo_only"] = True

        else:
            result = upload_results(
                out,
                result,
                job.output_s3_prefix,
            )

        result["total_function_seconds"] = round(
            time.perf_counter() - started,
            3,
        )

        return result
