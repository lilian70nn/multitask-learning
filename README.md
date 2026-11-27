# EndoVision AI

Interactive web workstation for GI endoscopy image and video analysis, lesion segmentation, evidential uncertainty estimation, clinical Q&A, and A4 PDF report export.

**Research demo. Not for clinical use.**

## Running it

This branch is set up for a single **NVIDIA RTX 4090 (24 GB)**: the vision and caption models run in fp16, the chat model (Qwen2.5-7B-Instruct) is loaded in 4-bit, and the app requires a login.

Follow **[`deploy/helena/DEPLOY_HELENA.md`](deploy/helena/DEPLOY_HELENA.md)** for requirements, installation (`deploy/helena/install.sh`), configuration (`.env`, see `deploy/helena/.env.example`), the smoke test, the systemd service and HTTPS.

Short version:

```bash
git clone -b duc/pretraining https://github.com/lilian70nn/multitask-learning.git
cd multitask-learning
bash deploy/helena/install.sh          # venv, pinned dependencies, weights, model caches, .env
# edit .env: set APP_USER and APP_PASSWORD
.venv/bin/python scripts/smoke_test_webapp.py --image /path/to/an/endoscopy/frame.jpg
.venv/bin/python server.py             # listens on 127.0.0.1:7860
```

## Repository layout

* `server.py`, `web_auth.py`: FastAPI backend (inference, streaming Q&A, PDF export, login)
* `static/`: React front end (before/after slider, uncertainty heatmap, threshold control, video keyframes); libraries are bundled in `static/vendor/`
* `requirements_web.txt`: pinned dependencies for the web app (`requirements_ui.txt` and `requirements.txt` are the original, unpinned lists)
* `scripts/`: weights download, smoke test, fp32/fp16 comparison
* `deploy/helena/`: deployment files and notes
* `pretraining/`, `classification_vit/`, `segmentation_vit/`, `captioning/`: model definitions and training code
