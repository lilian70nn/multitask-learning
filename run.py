
from pathlib import Path
from types import SimpleNamespace
import json

import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
from huggingface_hub import snapshot_download

from classification_vit.hierarchical import HierarchicalClassifier
from segmentation_vit.train_segmentation import ViTUNet
from caption.captioner_vit import load_captioning_model


REPO_ID = "Lian70/model_weights"

CLS_PATHS = {
    "ce": "classification/vit_gastronet_dino_ce/hierarchical",
    "edl": "classification/vit_gastronet_dino_edl/hierarchical",
}

SEG_PATHS = {
    "ce": "segmentation/vit_gastronet_dino_ce_colonbench/best_model.pt",
    "edl": "segmentation/vit_gastronet_dino_edl_colonbench/best_model.pt",
}

CAPTION_PATH = "decoder/dino_gastronet_hierarchical_colonbench"
VIT_PATH = "dino-weights/ema_backbone.pt"


def get_checkpoint_root(paths):
    return Path(snapshot_download(repo_id=REPO_ID, allow_patterns=paths))


class GastroNet:
    def __init__(self, task="classification", baseline=1, device=None):
        self.task = task.lower()
        self.baseline = baseline
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if self.task not in ("classification", "segmentation", "captioning"):
            raise ValueError("task must be classification, segmentation, or captioning")
        if self.task == "captioning" and self.baseline not in (1, 2, 3):
            raise ValueError("baseline must be 1, 2, or 3")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if self.task == "classification":
            self._load_classification()
        elif self.task == "segmentation":
            self._load_segmentation()
        else:
            self._load_captioning()

        self.model.to(self.device).eval()

    def _load_classification(self):
        cls_dir = CLS_PATHS["ce"]
        root = get_checkpoint_root([f"{cls_dir}/*"])
        self.model = HierarchicalClassifier(root / cls_dir, edl=False)

        with open(root / cls_dir / "groups.json") as f:
            groups = json.load(f)
        self.class_names = {int(k): v for k, v in groups["original_classes"].items()}

    def _load_segmentation(self):
        seg_path = SEG_PATHS["ce"]
        root = get_checkpoint_root([seg_path])
        state = torch.load(root / seg_path, map_location="cpu", weights_only=True)
        num_classes = state["decoder.5.weight"].shape[0]

        backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        self.model = ViTUNet(backbone, num_classes=num_classes)
        self.model.load_state_dict(state, strict=True)

    def _load_captioning(self):
        caption_dir = f"{CAPTION_PATH}/baseline_{self.baseline}"
        required = [VIT_PATH, f"{caption_dir}/adapter.pth", f"{caption_dir}/decoder/*"]
        cls_dir, seg_path = None, None

        if self.baseline >= 2:
            method = "edl" if self.baseline == 3 else "ce"
            cls_dir = CLS_PATHS[method]
            seg_path = SEG_PATHS[method]
            required.extend([f"{cls_dir}/*", seg_path])

        root = get_checkpoint_root(required)

        args = SimpleNamespace(
            baseline=self.baseline,
            vit_ckpt=str(root / VIT_PATH),
            cls_ckpt=str(root / cls_dir) if cls_dir else None,
            seg_ckpt=str(root / seg_path) if seg_path else None,
            adapter_ckpt=str(root / caption_dir / "adapter.pth"),
            qwen_name=str(root / caption_dir / "decoder"),
            seg_grid_size=8,
        )

        self.model = load_captioning_model(args)

    def __call__(self, image):
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                image = img.convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError("image must be a file path or PIL Image")

        original_size = image.size
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            if self.task == "classification":
                logits = self.model(x)
                probs = F.softmax(logits, dim=1)
                index = int(probs.argmax(dim=1).item())

                return {
                    "class_index": index,
                    "class_name": self.class_names[index],
                    "confidence": float(probs[0, index].item()),
                    "probabilities": probs[0].cpu().tolist(),
                }

            if self.task == "segmentation":
                logits = self.model(x)
                logits = F.interpolate(logits, size=(original_size[1], original_size[0]), mode="bilinear", align_corners=False)
                return logits.argmax(dim=1)[0].cpu().numpy()

            if self.task == "captioning":
                return self.model.generate(x)[0]
