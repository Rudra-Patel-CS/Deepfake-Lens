# Deepfake Lens

Smart India Hackathon project for detecting face-swap deepfake videos with spatial, frequency, and temporal analysis.


## Quick start from this repository

The `main` branch includes the web UI, API, trained `full_v3` inference weights,
configuration, evaluation report, and Render deployment files.

For Render, connect **Rudra-Patel-CS/Deepfake-Lens**, choose branch **main**, and
use the included `render.yaml` Blueprint. For manual setup, follow the Render
section below, including its environment variables. Leave Root Directory empty.

For a fresh local checkout, after creating a virtual environment and installing
`requirements.txt`, install the bundled model assets and start the app:

```powershell
.\venv\Scripts\python.exe -m scripts.prepare_render
.\start-web.bat
```

Open http://127.0.0.1:8000. No training dataset or retraining is required for
prediction. `prepare_render` copies the bundled inference checkpoint; use it for
a fresh installation, not to replace an active training checkpoint.

## Implemented

- OpenCV video frame extraction.
- RetinaFace face detection and largest-face cropping.
- FFT magnitude images for cropped faces.
- Video-level train/validation/test metadata splits.
- RGB + FFT sequence dataset: `(rgb_seq, fft_seq, label)`.
- Three-branch PyTorch model:
  - EfficientNet-B0 frame branch: 1280 features.
  - CNN frequency branch: 128 features.
  - BiLSTM temporal branch: 256 features.
  - Fusion classifier with auxiliary branch heads.
- Training, ablations, checkpoints, metrics, ROC, confusion matrix, and failure reports.
- Gradient heatmaps, FFT heatmaps, and optional SHAP explanations.

## Structure

```text
ai/models/          Model and dataset
ai/preprocessing/   Frames, faces, FFT, metadata
ai/inference/       Video score aggregation
ai/explain/         Heatmaps and branch explanations
scripts/            Preprocess, train, evaluate, explain
 tests/             Pytest tests
dataset/raw/        Input videos, not committed
dataset/processed/ Generated data, not committed
trained_models/     Checkpoints, not committed
reports/            Metrics and explanation outputs
```

## Setup

Windows PowerShell, Python 3.13:

```powershell
git clone https://github.com/Rudra-Patel-CS/Deepfake-Lens.git
cd Deepfake-Lens
py -3.13 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m scripts.setup
```

After future updates:

```powershell
git pull
venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Place videos here:

```text
dataset/raw/ffpp_real/*.mp4
dataset/raw/ffpp_fake/*.mp4
```

Real files use names such as `000.mp4`. Fake FaceForensics++ files use names such as `000_003.mp4`.

## Run Pipeline

1. Preprocess a small sample first:

```powershell
python -m scripts.preprocess --limit 5
```

This creates frames, cropped faces, FFT images, and `dataset/processed/metadata.csv`. Face detection can be slow on CPU.

2. Inspect data and run tests:

```powershell
python -m scripts.inspect_dataset
python -m pytest -q
```

Run this from the `Deepfake-Video-detector-main` directory after activating its
`venv`. Using a system-wide `pytest` (or running tests from the parent folder)
will not use the project's installed PyTorch dependencies.

3. Train the full model:

```powershell
python -m scripts.train --epochs 10 --batch-size 4 --lr 1e-4 --branches full --seq-len 10 --run-name full_model
```

CPU smoke training:

```powershell
python -m scripts.train --epochs 2 --batch-size 4 --branches full --seq-len 10 --limit 5 --run-name cpu_smoke
```

4. Run ablations:

```powershell
python -m scripts.ablation --epochs 5 --batch-size 4 --seq-len 10 --limit 5
```

Configurations: `frame`, `frequency`, `frame+temporal`, and `full`.

5. Evaluate the held-out test split:

```powershell
python -m scripts.evaluate --run-name full_model
```

6. Explain one video:

```powershell
python -m scripts.explain --video dataset/raw/ffpp_real/000.mp4 --run-name full_model
```

## Outputs

```text
dataset/processed/metadata.csv
trained_models/<run>/best.pt
trained_models/<run>/last.pt
trained_models/<run>/config.json
reports/<run>/history.csv
reports/<run>/test_metrics.json
reports/<run>/roc_curve.png
reports/<run>/confusion_matrix.png
reports/<run>/failure_cases.json
reports/ablation_val.csv
reports/explanations/<video>/
```

## Web Application

The project includes a local full-stack upload application. The browser UI is in
`frontend/`; its FastAPI backend loads `trained_models/full_v3/best.pt`, runs
face detection and model inference, and returns the verdict, confidence,
Grad-CAM detected frames, and branch explanation.

```powershell
venv\Scripts\python.exe -m uvicorn ai.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` and upload an MP4, MOV, AVI, or WebM video.
Generated explanation assets are stored in `reports/web_analyses/`.

Training uses only train/validation data. The test split is reserved for final evaluation. Runtime settings such as `FRAME_INTERVAL`, `IMG_SIZE`, and performance thresholds are in `ai/utils/config.py`.

PyTorch inference and Grad-CAM use `cuda:0` automatically when CUDA is available. On native Windows, the TensorFlow runtime used by RetinaFace does not support CUDA in current TensorFlow releases, so face extraction remains CPU-bound; use WSL2 or TensorFlow-DirectML if GPU face detection is required.

## Current Limitations

- The frontend is a responsive single-page HTML interface served by FastAPI; it is not a separate React application.
- CPU RetinaFace processing is slow.
- SHAP uses a fallback attribution method when the optional package is unavailable.
- Evaluation metrics are unreliable when only a few test videos are available.


## full preprocessing
- .\venv\Scripts\python.exe -m scripts.preprocess

## Running the imported full_v3 model on this PC

The files supplied on 20 November 2026 are installed at:

- `trained_models/full_v3/best.pt` (the supplied `best.pt.zip` is already a PyTorch checkpoint; its bytes were copied unchanged)
- `trained_models/full_v3/config.json`
- `reports/full_v3/test_metrics.json`

Run `start-web.bat`, then open http://127.0.0.1:8000. This checkout is based on
`origin/final` and serves the original `frontend/index.html` UI with the supplied
model. Stop the terminal with Ctrl+C.

The supplied evaluation reports 78% accuracy on 300 videos (234 correct), F1
0.8024 and ROC-AUC 0.8930. Its operating threshold is 0.006866547744721174;
it must not be replaced by 0.5. The report has not been independently reproduced
on this PC, and a video's score is not accuracy or calibrated confidence.

This PC defaults to OpenCV face detection because Windows Application Control
previously blocked a TensorFlow DLL used by RetinaFace. OpenCV changes face
crops, so the supplied evaluation does not establish accuracy for this fallback.
On a working RetinaFace environment, set `FACE_DETECTOR=retinaface` before
starting the server to use the original final-branch extraction path.

The previous local edits are preserved in a Git stash named
`Preserve local setup before full_v3 integration` and in
`temp/pre-full-v3-backup/`. Existing datasets and checkpoints were retained.

## Render free deployment (experimental)

Select GitHub branch `main`, Python runtime, and the Free instance.
Use `.python-version` (Python 3.11); leave Root Directory empty.

Build command:

```sh
pip install -r requirements-render.txt && python -m scripts.prepare_render
```

Start command:

```sh
uvicorn ai.api:app --host 0.0.0.0 --port $PORT --workers 1
```

Set `RENDER_LIGHTWEIGHT=1`, `FACE_DETECTOR=opencv`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and `MALLOC_ARENA_MAX=2`. Set health check `/api/health`.
Alternatively use the included `render.yaml` Blueprint, which sets these values.

`deploy/full_v3/best.pt` contains the original learned weights and configuration
with optimizer/training history removed. The build copies it and the supplied
metrics into the runtime directories; no model download link or retraining is
required. Your original local training checkpoint remains unchanged.

Free mode uses CPU-only PyTorch, processes one frame at a time inside each spatial
branch, limits videos to 20 MB / 60 seconds, and accepts one analysis at a time.
It keeps the same temporal sequence and supplied decision threshold. Grad-CAM,
SHAP and attribution charts are disabled to reduce memory; prediction and JSON
reports remain available. The UI identifies this mode. Reports are temporary and
are lost when Render restarts or spins down.

Render Free has only 512 MB RAM and 0.1 CPU. This setup is an experimental attempt,
not a guarantee that video inference fits that limit. CPU inference may be slow,
and an out-of-memory restart can still require a larger instance or a smaller
model. Free services sleep after 15 minutes without traffic. Local Windows tests
do not reproduce Render's Linux memory cap; verify the deployed logs before
relying on this service.
