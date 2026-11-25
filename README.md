# EndoVision AI 

Interactive web workstation for GI endoscopy image and video analysis, lesion segmentation, evidential uncertainty estimation, clinical Q&A, and A4 PDF report export.

## Prerequisites

* **GPU:** NVIDIA CUDA GPU with $\ge 24\text{ GB}$ VRAM (e.g., RTX A6000, L40S, A100)
* **Python:** 3.10+
* **Network:** Expose HTTP port **`7860`** (a remote server)

## Quick Start

### 1. Clone the `ui` branch
```bash
git clone -b ui https://github.com/lilian70nn/multitask-learning.git
cd multitask-learning
```

### 2. Install dependencies
```bash
pip install -r requirements_ui.txt
```

### 3. Download pretrained weights from Hugging Face
```bash
python3 -c '
from huggingface_hub import snapshot_download
import os, shutil

target = "./model_weights"
snapshot_download(repo_id="Lian70/model_weights", local_dir=target, repo_type="model")

nested = os.path.join(target, "model_weights")
if os.path.isdir(nested):
    for item in os.listdir(nested):
        shutil.move(os.path.join(nested, item), os.path.join(target, item))
    os.rmdir(nested)
print("Model weights ready!")
'
```

### 4. Start the server
```bash
python3 server.py
```

* **Local machine:** Open `http://localhost:7860`
* **RunPod:** Click **Connect → Connect to HTTP Service [Port 7860]** (or open `https://<POD_ID>-7860.proxy.runpod.net`)

## Repository Layout (`ui` branch)

* `server.py` — FastAPI backend serving perception inference, streaming clinical Q&A, and A4 PDF report generation
* `static/index.html` — Single-page React frontend (Before/After wipe slider, uncertainty heatmap, live threshold control, video keyframe inspection)
* `requirements_ui.txt` — Python dependencies for the web server and inference pipeline
* `pretraining/`, `classification_vit/`, `segmentation_vit/`, `captioning/` — Core model definitions and training scripts
