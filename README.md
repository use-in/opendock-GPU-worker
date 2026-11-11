# OPENDOCK Boltz-2 GPU Worker

OpenDock의 AI 기반 단백질–리간드 구조 예측을 수행하는
Modal GPU Worker입니다.

## Core Technologies

- Boltz-2 2.2.1
- NVIDIA H100
- Modal
- PyTorch Lightning
- AWS S3
- RDKit

## Architecture

Backend / AWS Step Functions
        |
        v
Modal submit() - CPU routing
        |
        +--> Small Worker  (H100)
        |
        +--> Medium Worker (H100)
        |
        +--> Large Worker  (H100)
                     |
                     v
             Resident Boltz-2
                     |
                     v
                AWS S3

## Model Caching

1. Model weights are cached in a persistent Modal Volume.
2. Boltz-2 is loaded once when a GPU container starts.
3. Subsequent requests reuse the same Python model object.
4. Each request may specify its own sampling options.

## Input

- Job ID and attempt
- Protein sequence
- Ligand SMILES
- MSA S3 URI
- Output S3 prefix
- Inference options

Production requests require an external MSA.

## Output

- Predicted structure (.cif)
- Confidence JSON
- Additional prediction artifacts (.npz)
- Manifest JSON

The manifest is uploaded after all result artifacts.

## Routing

Routing estimate:

ceil(1.2 * (protein residues + ligand heavy atoms))

- Small: <= 1024
- Medium: <= 2560
- Large: <= 5120

All pools use H100 GPUs.

Thresholds are provisional and require VRAM benchmarking.

## Known Experimental Results

Input:
- Ubiquitin (76 residues)
- Ethanol (CCO)
- MSA: empty
- Recycling steps: 1
- Sampling steps: 20
- Diffusion samples: 1

Results:
- CLI second run: 30.490 seconds
- Resident model second request: 4.756 seconds
- Resident model initial load: 26.144 seconds

These are experimental results, not production benchmarks.

## Development

Install:

    py -m pip install -r requirements-local.txt

Test:

    py -m pytest -q

Smoke test:

    py -m modal run modal_worker.py::smoke

Deploy:

    py -m modal deploy modal_worker.py

Submit:

    py invoke_deployed.py job.json

## Security

- Never commit cloud credentials.
- Never commit model checkpoints.
- Use Modal Secrets for S3 configuration.
- Restrict S3 permissions to approved buckets.
- Backend is responsible for user authentication.

## Current Limitations

- S3 production integration has not yet been verified.
- Routing thresholds are not VRAM-calibrated.
- Affinity prediction is not implemented.
- Worker-level idempotency and retry locking are pending.
- Joltz is researched separately and excluded here.
