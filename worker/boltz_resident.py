
"""Boltz-2 2.2.1 in-process model caching.

No CLI subprocess is started for each request.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

from worker.contracts import Options


def load_model(cache: Path):
    """Load checkpoint once per Modal GPU container."""

    import torch

    from boltz.main import (
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
        download_boltz2,
    )

    from boltz.model.models.boltz2 import Boltz2

    started = time.perf_counter()

    cache.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Reuses downloaded checkpoint files when present.
    download_boltz2(cache)

    defaults = Options()

    model = Boltz2.load_from_checkpoint(
        cache / "boltz2_conf.ckpt",
        strict=True,
        map_location="cpu",
        ema=False,
        use_kernels=True,
        predict_args={
            "recycling_steps": defaults.recycling_steps,
            "sampling_steps": defaults.sampling_steps,
            "diffusion_samples": defaults.diffusion_samples,
            "max_parallel_samples": None,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
        },
        diffusion_process_args=asdict(
            Boltz2DiffusionParams()
        ),
        pairformer_args=asdict(
            PairformerArgsV2()
        ),
        msa_args=asdict(
            MSAModuleArgs(
                subsample_msa=False,
                use_paired_feature=True,
            )
        ),
        steering_args=asdict(
            BoltzSteeringParams()
        ),
    )

    model.eval().to("cuda")
    torch.cuda.synchronize()

    load_seconds = round(
        time.perf_counter() - started,
        3,
    )

    return model, load_seconds


def predict_with_model(
    model,
    yaml_path: Path,
    output_dir: Path,
    cache_dir: Path,
    options: Options,
) -> dict:
    """Run a single prediction with an existing model."""

    import torch

    from rdkit import Chem
    from pytorch_lightning import Trainer

    from boltz.main import process_inputs
    from boltz.data.types import Manifest

    from boltz.data.module.inferencev2 import (
        Boltz2InferenceDataModule,
    )

    from boltz.data.write.writer import BoltzWriter

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    Chem.SetDefaultPickleProperties(
        Chem.PropertyPickleOptions.AllProps
    )

    # ========================================
    # Step 1: Preprocessing
    # ========================================

    t = time.perf_counter()

    process_inputs(
        data=[yaml_path],
        out_dir=output_dir,
        ccd_path=cache_dir / "ccd.pkl",
        mol_dir=cache_dir / "mols",
        use_msa_server=False,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        msa_server_username=None,
        msa_server_password=None,
        api_key_header=None,
        api_key_value=None,
        boltz2=True,
        preprocessing_threads=2,
        max_msa_seqs=8192,
    )

    processed = output_dir / "processed"

    manifest = Manifest.load(
        processed / "manifest.json"
    )

    if len(manifest.records) != 1:
        raise RuntimeError(
            "Expected one Boltz input"
        )

    # ========================================
    # Step 2: Output Writer
    # ========================================

    writer = BoltzWriter(
        data_dir=processed / "structures",
        output_dir=output_dir / "predictions",
        output_format="mmcif",
        boltz2=True,
    )

    # ========================================
    # Step 3: Inference DataModule
    # ========================================

    data = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=processed / "structures",
        msa_dir=processed / "msa",
        mol_dir=cache_dir / "mols",
        num_workers=0,
        constraints_dir=(
            processed / "constraints"
            if (processed / "constraints").exists()
            else None
        ),
        template_dir=(
            processed / "templates"
            if (processed / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            processed / "mols"
            if (processed / "mols").exists()
            else None
        ),
        override_method=None,
    )

    prep_seconds = time.perf_counter() - t

    # ========================================
    # Step 4: Update Job Options
    # ========================================

    # Boltz2.predict_step reads model.predict_args.
    # This avoids reloading the checkpoint per job.

    model.predict_args.update({
        "recycling_steps": options.recycling_steps,
        "sampling_steps": options.sampling_steps,
        "diffusion_samples": options.diffusion_samples,
    })

    # ========================================
    # Step 5: PyTorch Lightning Prediction
    # ========================================

    trainer = Trainer(
        default_root_dir=str(output_dir),
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        callbacks=[writer],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    torch.cuda.reset_peak_memory_stats()

    t = time.perf_counter()

    try:
        trainer.predict(
            model,
            datamodule=data,
            return_predictions=False,
        )

        torch.cuda.synchronize()

        predict_seconds = (
            time.perf_counter() - t
        )

    finally:
        # Lightning moves the model to CPU at teardown.
        # Restore it for the next request.
        model.eval().to("cuda")
        torch.cuda.synchronize()

    return {
        "preprocessing_seconds": round(
            prep_seconds,
            3,
        ),
        "inference_seconds": round(
            predict_seconds,
            3,
        ),
        "gpu_memory_peak_mib": round(
            torch.cuda.max_memory_allocated()
            / 1048576,
            1,
        ),
    }
