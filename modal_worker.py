
"""OpenDock: three independent H100 pools with resident Boltz-2 models.

Deploy: py -m modal deploy modal_worker.py
Smoke:  py -m modal run modal_worker.py::smoke
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "opendock-boltz2-worker"
MODEL_CACHE_DIR = Path("/model-cache/boltz")

PROTEIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLED"
    "GRTLSDYNIQKESTLHLVLRLRGG"
)

app = modal.App(APP_NAME)


# GPU image: Boltz-2 inference
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgomp1", "libgl1")
    .pip_install(
        "boltz[cuda]==2.2.1",
        "boto3>=1.34,<2",
        "PyYAML>=6,<7",
    )
    .add_local_python_source("worker")
)


# CPU image: validate job and select pool
router_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "rdkit>=2024.9",
        "boto3>=1.34,<2",
    )
    .add_local_python_source("worker")
)


# Persistent checkpoint cache
model_volume = modal.Volume.from_name(
    "opendock-boltz-model-cache",
    create_if_missing=True,
)


# Must be configured in Modal settings.
s3_secret = modal.Secret.from_name("opendock-s3")


def _startup(instance):
    """Load Boltz-2 once per Modal GPU container."""

    import torch
    from worker.boltz_resident import load_model

    torch.set_float32_matmul_precision("highest")

    instance.model, instance.load_seconds = load_model(
        MODEL_CACHE_DIR
    )

    instance.request_count = 0

    model_volume.commit()

    print(
        "[MODEL] cached model ready; "
        f"load={instance.load_seconds:.2f}s",
        flush=True,
    )


def _predict(
    instance,
    payload: dict,
    pool: str,
) -> dict:
    """Execute one job using the already loaded model."""

    from worker.contracts import parse_job
    from worker.routing import route_job
    from worker.pipeline import execute

    job = parse_job(payload)

    actual_pool = route_job(job)["pool"]

    if pool != actual_pool:
        raise ValueError(
            f"Wrong pool: expected {actual_pool}, "
            f"received {pool}"
        )

    instance.request_count += 1

    result = execute(
        payload,
        model=instance.model,
        cache_dir=MODEL_CACHE_DIR,
    )

    result["pool"] = pool
    result["gpu"] = "H100"

    result["container_request_count"] = (
        instance.request_count
    )

    result["model_loaded_once_seconds"] = (
        instance.load_seconds
    )

    return result


# ============================================================
# Small Pool
# ============================================================

@app.cls(
    image=gpu_image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=3600,
    max_containers=1,
    min_containers=0,
    scaledown_window=300,
    retries=0,
    secrets=[s3_secret],
    volumes={
        str(MODEL_CACHE_DIR.parent): model_volume
    },
)
class SmallWorker:

    @modal.enter()
    def enter(self):
        _startup(self)

    @modal.method()
    def predict(self, payload: dict) -> dict:
        return _predict(self, payload, "small")

    @modal.method()
    def smoke(self) -> dict:
        """Smoke test using empty MSA; no S3 upload."""

        from worker.pipeline import execute

        payload = {
            "job_id": "smoke-ubiquitin",
            "attempt": 1,
            "protein_sequence": PROTEIN,
            "ligand_smiles": "CCO",
            "msa": {
                "mode": "empty"
            },
            "options": {
                "recycling_steps": 1,
                "sampling_steps": 20,
                "diffusion_samples": 1,
            },
        }

        self.request_count += 1

        result = execute(
            payload,
            model=self.model,
            cache_dir=MODEL_CACHE_DIR,
            demo=True,
        )

        result["container_request_count"] = (
            self.request_count
        )

        result["model_loaded_once_seconds"] = (
            self.load_seconds
        )

        return result


# ============================================================
# Medium Pool
# ============================================================

@app.cls(
    image=gpu_image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=3600,
    max_containers=1,
    min_containers=0,
    scaledown_window=300,
    retries=0,
    secrets=[s3_secret],
    volumes={
        str(MODEL_CACHE_DIR.parent): model_volume
    },
)
class MediumWorker:

    @modal.enter()
    def enter(self):
        _startup(self)

    @modal.method()
    def predict(self, payload: dict) -> dict:
        return _predict(self, payload, "medium")


# ============================================================
# Large Pool
# ============================================================

@app.cls(
    image=gpu_image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=3600,
    max_containers=1,
    min_containers=0,
    scaledown_window=300,
    retries=0,
    secrets=[s3_secret],
    volumes={
        str(MODEL_CACHE_DIR.parent): model_volume
    },
)
class LargeWorker:

    @modal.enter()
    def enter(self):
        _startup(self)

    @modal.method()
    def predict(self, payload: dict) -> dict:
        return _predict(self, payload, "large")


# ============================================================
# Authenticated Backend Entry Point
# ============================================================

@app.function(
    image=router_image,
    cpu=1,
    memory=1024,
    timeout=3600,
    retries=0,
    max_containers=3,
)
def submit(payload: dict) -> dict:
    """Route a job to the selected H100 Worker Pool."""

    from worker.contracts import parse_job
    from worker.routing import route_job

    decision = route_job(
        parse_job(payload)
    )

    pool = decision["pool"]

    worker = {
        "small": SmallWorker,
        "medium": MediumWorker,
        "large": LargeWorker,
    }[pool]()

    result = worker.predict.remote(payload)

    return {
        "routing": decision,
        "result": result,
    }


# ============================================================
# Local Smoke Test
# ============================================================

@app.local_entrypoint()
def smoke():

    worker = SmallWorker()

    first = worker.smoke.remote()
    second = worker.smoke.remote()

    print(
        json.dumps(
            {
                "first": first,
                "second": second,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
