
"""S3-free Modal H100 smoke test for a resident Boltz-2 model."""

from pathlib import Path
import json
import modal

app = modal.App("opendock-boltz2-smoke")
MODEL_CACHE_DIR = Path("/model-cache/boltz")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgomp1", "libgl1")
    .pip_install(
        "boltz[cuda]==2.2.1",
        "boto3>=1.34,<2",
        "PyYAML>=6,<7",
    )
    .add_local_python_source("worker")
)

model_volume = modal.Volume.from_name(
    "opendock-boltz-model-cache",
    create_if_missing=True,
)

PROTEIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLED"
    "GRTLSDYNIQKESTLHLVLRLRGG"
)


@app.cls(
    image=image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=1800,
    max_containers=1,
    min_containers=0,
    scaledown_window=300,
    retries=0,
    volumes={
        str(MODEL_CACHE_DIR.parent): model_volume
    },
)
class SmokeWorker:

    @modal.enter()
    def load_once(self):
        from worker.boltz_resident import load_model

        self.model, self.load_seconds = load_model(
            MODEL_CACHE_DIR
        )
        self.request_count = 0
        model_volume.commit()

    @modal.method()
    def predict(self):
        from worker.pipeline import execute

        self.request_count += 1

        payload = {
            "job_id": "smoke-ubiquitin",
            "attempt": 1,
            "protein_sequence": PROTEIN,
            "ligand_smiles": "CCO",
            "msa": {"mode": "empty"},
            "options": {
                "recycling_steps": 1,
                "sampling_steps": 20,
                "diffusion_samples": 1,
            },
        }

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


@app.local_entrypoint()
def smoke():
    worker = SmokeWorker()

    first = worker.predict.remote()
    second = worker.predict.remote()

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
