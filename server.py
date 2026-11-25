import os, re, io, sys, gc, uuid, json, base64, textwrap, threading, subprocess
from datetime import datetime
from types import SimpleNamespace
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, SwinForImageClassification, TextIteratorStreamer

_orig_swin_load = SwinForImageClassification.load_state_dict
def _compat_swin_load(self, state_dict, strict=True, assign=False):
    if any(".attention.self.query." in k or ".intermediate.dense." in k for k in state_dict):
        model_keys = set(self.state_dict().keys())
        remapped = {}
        for k, v in state_dict.items():
            nk = (
                k.replace(".attention.self.query.", ".attention.q_proj.")
                 .replace(".attention.self.key.", ".attention.k_proj.")
                 .replace(".attention.self.value.", ".attention.v_proj.")
                 .replace(".attention.output.dense.", ".attention.o_proj.")
                 .replace(".attention.self.relative_position_bias_table", ".attention.relative_position_bias.relative_position_bias_table")
                 .replace(".attention.self.relative_position_index", ".attention.relative_position_bias.relative_position_index")
                 .replace(".intermediate.dense.", ".mlp.fc1.")
                 .replace(".output.dense.", ".mlp.fc2.")
            )
            if nk in model_keys or not strict:
                remapped[nk] = v
        state_dict = remapped
    return _orig_swin_load(self, state_dict, strict=strict, assign=assign)

SwinForImageClassification.load_state_dict = _compat_swin_load

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MULTITASK_REPO = BASE_DIR if os.path.isdir(os.path.join(BASE_DIR, "captioning")) else "/workspace/multitask-learning"
if MULTITASK_REPO not in sys.path:
    sys.path.insert(0, MULTITASK_REPO)

from captioning.captioner_vit import load_captioning_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = os.path.join(BASE_DIR, "model_weights") if os.path.isdir(os.path.join(BASE_DIR, "model_weights")) else "/workspace/model_weights"
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CLASS_NAMES = [
    "bbps-0-1", "bbps-2-3", "cecum", "dyed-lifted-polyps",
    "dyed-resection-margins", "esophagitis-a", "esophagitis-b-d",
    "impacted-stool", "polyps", "pylorus", "retroflex-rectum",
    "retroflex-stomach", "ulcerative-colitis-grade-1",
    "ulcerative-colitis-grade-2", "ulcerative-colitis-grade-3", "z-line",
]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VLM_CACHE, CHAT_LLM_CACHE, SESSION_CACHE = {}, {}, {}
IMG_GEN_PIPE, VID_GEN_PIPE = None, None

app = FastAPI(title="EndoVision AI Production Platform", version="2.0.0")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

def np_to_b64(arr: np.ndarray, fmt: str = "PNG") -> str:
    pil_img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def b64_to_np(data_url: str) -> Optional[np.ndarray]:
    if not data_url:
        return None
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return np.array(Image.open(io.BytesIO(base64.b64decode(data_url))).convert("RGB"))

def get_vlm_model(seg_domain: str = "colonbench"):
    if seg_domain in VLM_CACHE:
        VLM_CACHE[seg_domain].to(DEVICE)
        return VLM_CACHE[seg_domain]
    dom = "polyp" if seg_domain == "polyp" else "colonbench"
    args = SimpleNamespace(
        baseline=3,
        vit_ckpt=f"{WEIGHTS_DIR}/dino-weights/ema_backbone.pt",
        cls_ckpt=f"{WEIGHTS_DIR}/classification/vit_gastronet_dino_edl/hierarchical",
        seg_ckpt=f"{WEIGHTS_DIR}/segmentation/vit_gastronet_dino_edl_{dom}/best_model.pt",
        adapter_ckpt=f"{WEIGHTS_DIR}/decoder/dino_gastronet_hierarchical_{dom}/baseline_3/adapter.pth",
        qwen_name=f"{WEIGHTS_DIR}/decoder/dino_gastronet_hierarchical_{dom}/baseline_3/decoder",
        seg_grid_size=8,
    )
    model = load_captioning_model(args).to(DEVICE).eval()
    VLM_CACHE[seg_domain] = model
    return model

def get_chat_llm(model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
    target_dtype = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
    if model_id not in CHAT_LLM_CACHE:
        CHAT_LLM_CACHE.clear()
        gc.collect()
        torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=target_dtype).to(device=DEVICE, dtype=target_dtype).eval()
        CHAT_LLM_CACHE[model_id] = (tokenizer, model)
    else:
        tokenizer, model = CHAT_LLM_CACHE[model_id]
        model.to(device=DEVICE, dtype=target_dtype).eval()
    return tokenizer, model

def offload_reasoning_models():
    for m in VLM_CACHE.values():
        m.to("cpu")
    for _, m in CHAT_LLM_CACHE.values():
        m.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

def describe_mask_location(binary_mask: np.ndarray) -> str:
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return "No focal lesion"
    cx, cy = xs.mean() / binary_mask.shape[1], ys.mean() / binary_mask.shape[0]
    vert = "Upper" if cy < 0.4 else ("Lower" if cy > 0.6 else "Central")
    horiz = "left" if cx < 0.4 else ("right" if cx > 0.6 else "midline")
    return f"{vert}-{horiz} quadrant"

def render_overlay_from_prob(rgb: np.ndarray, polyp_prob: np.ndarray, mask_thresh: float):
    raw_mask = (polyp_prob >= mask_thresh).astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    raw_mask = raw_mask & (hsv[:, :, 2] >= 45).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    binary_mask = np.zeros_like(raw_mask)
    for lbl_id in range(1, num_labels):
        if stats[lbl_id, cv2.CC_STAT_AREA] >= 75:
            binary_mask[labels == lbl_id] = 1
    overlay = rgb.copy()
    overlay[binary_mask == 1] = [244, 63, 94]
    blended = cv2.addWeighted(rgb, 0.68, overlay, 0.32, 0)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (45, 212, 191), 2)
    mask_512 = cv2.resize(binary_mask * 255, (512, 512), interpolation=cv2.INTER_NEAREST)
    return blended, np.stack([mask_512] * 3, axis=-1), float(binary_mask.mean() * 100.0), describe_mask_location(binary_mask)

@torch.inference_mode()
def analyze_single_tensor(model, tensor: torch.Tensor, pil_resized: Image.Image, mask_thresh: float = 0.5):
    cls_logits = model.cls_model(tensor)
    seg_logits = model.seg_model(tensor)
    cls_alpha = F.softplus(cls_logits) + 1.0
    cls_probs = cls_alpha / cls_alpha.sum(dim=-1, keepdim=True)
    cls_uncertainty = float((len(CLASS_NAMES) / cls_alpha.sum(dim=-1)).item())
    top3_probs, top3_idxs = torch.topk(cls_probs[0], k=3)
    top3_list = [{"label": CLASS_NAMES[int(i.item())], "prob": float(p.item())} for p, i in zip(top3_probs, top3_idxs)]
    seg_alpha = F.softplus(seg_logits) + 1.0
    seg_probs = seg_alpha / seg_alpha.sum(dim=1, keepdim=True)
    polyp_prob = seg_probs[0, 1].cpu().numpy()
    seg_uncertainty = (2.0 / seg_alpha.sum(dim=1))[0].cpu().numpy()
    rgb = np.array(pil_resized)
    blended, mask_512_rgb, area_pct, location = render_overlay_from_prob(rgb, polyp_prob, mask_thresh)
    unc_norm = np.clip(seg_uncertainty * 255.0, 0, 255).astype(np.uint8)
    unc_heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(unc_norm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return {
        "rgb": rgb, "polyp_prob": polyp_prob, "blended": blended, "binary_mask_rgb": mask_512_rgb,
        "uncertainty_map": unc_heatmap_rgb, "pred_label": top3_list[0]["label"], "pred_conf": top3_list[0]["prob"],
        "top3": top3_list, "cls_uncertainty": cls_uncertainty, "polyp_area_pct": area_pct,
        "peak_polyp_prob": float(polyp_prob.max()), "location": location,
    }

@app.get("/", response_class=HTMLResponse)
async def serve_spa():
    with open(os.path.join(BASE_DIR, "static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/health")
async def health_check():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_used = round(torch.cuda.memory_allocated(0) / (1024**3), 1) if torch.cuda.is_available() else 0.0
    vram_total = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1) if torch.cuda.is_available() else 0.0
    return {"gpu": gpu_name, "vram_used_gb": vram_used, "vram_total_gb": vram_total}

@app.post("/api/analyze")
async def api_analyze(
    file: UploadFile = File(...), seg_domain: str = Form("colonbench"),
    mask_thresh: float = Form(0.5), sample_fps: int = Form(2),
):
    model = get_vlm_model(seg_domain)
    session_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename or "")[1].lower()
    raw_bytes = await file.read()
    is_video = ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"] or (file.content_type or "").startswith("video")

    if not is_video:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        pil_resized = pil_img.resize((224, 224))
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            res = analyze_single_tensor(model, tensor, pil_resized, mask_thresh)
            caption = model.generate(tensor, max_new_tokens=64)[0].strip()
        top3_str = ", ".join([f"{x['label']} ({x['prob']*100:.1f}%)" for x in res["top3"]])
        structured_context = (
            f"MODALITY: Single Endoscopy Frame ({seg_domain} checkpoint)\n"
            f"PRIMARY CLASSIFICATION: {res['pred_label']} ({res['pred_conf']*100:.1f}% confidence)\n"
            f"TOP-3 DIFFERENTIAL: {top3_str}\n"
            f"UNCERTAINTY: {res['cls_uncertainty']:.3f}\n"
            f"MASK COVERAGE: {res['polyp_area_pct']:.2f}% of frame (peak prob: {res['peak_polyp_prob']:.2f})\n"
            f"LESION SPATIAL LOCATION: {res['location']}\n"
            f"CAPTION: {caption}"
        )
        SESSION_CACHE[session_id] = {
            "mode": "image", "rgb": res["rgb"], "polyp_prob": res["polyp_prob"],
            "uncertainty_map": res["uncertainty_map"], "pred_label": res["pred_label"],
            "pred_conf": res["pred_conf"], "top3": res["top3"], "cls_uncertainty": res["cls_uncertainty"],
            "caption": caption, "structured_context": structured_context,
        }
        return {
            "session_id": session_id, "mode": "image", "raw_image": np_to_b64(res["rgb"]),
            "overlay_image": np_to_b64(res["blended"]), "uncertainty_image": np_to_b64(res["uncertainty_map"]),
            "mask_image": np_to_b64(res["binary_mask_rgb"]), "video_url": None, "keyframes": [],
            "metrics": {
                "finding": res["pred_label"], "confidence": res["pred_conf"], "uncertainty": res["cls_uncertainty"],
                "coverage": res["polyp_area_pct"], "location": res["location"], "top3": res["top3"],
                "caption": caption, "structured_context": structured_context,
            },
        }

    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    in_path = f"{OUTPUTS_DIR}/in_{session_id}{ext or '.mp4'}"
    clean_in_mp4 = f"{OUTPUTS_DIR}/clean_{session_id}.mp4"
    with open(in_path, "wb") as f:
        f.write(raw_bytes)
    subprocess.run([ffmpeg_bin, "-y", "-i", in_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", clean_in_mp4], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    src_video = clean_in_mp4 if os.path.exists(clean_in_mp4) and os.path.getsize(clean_in_mp4) > 0 else in_path
    cap = cv2.VideoCapture(src_video)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(orig_fps / sample_fps)))
    raw_out, h264_out = f"{OUTPUTS_DIR}/raw_ov_{session_id}.mp4", f"{OUTPUTS_DIR}/overlay_{session_id}.mp4"
    writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), sample_fps, (224, 224))
    frame_records, frame_idx, ema_probs = [], 0, None

    with torch.inference_mode():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                timestamp = frame_idx / orig_fps
                pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                pil_resized = pil_img.resize((224, 224))
                rgb_arr = np.array(pil_resized)
                gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                r_m, g_m, b_m = rgb_arr[:, :, 0].mean(), rgb_arr[:, :, 1].mean(), rgb_arr[:, :, 2].mean()
                is_uninformative = (sharpness < 32.0) or (float(r_m / (g_m + b_m + 1.0)) > 2.3 and sharpness < 55.0)
                tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
                res = analyze_single_tensor(model, tensor, pil_resized, mask_thresh)
                if is_uninformative:
                    res["polyp_prob"] = np.zeros_like(res["polyp_prob"])
                    res["blended"] = rgb_arr.copy()
                    res["binary_mask_rgb"] = np.zeros((512, 512, 3), dtype=np.uint8)
                    res["polyp_area_pct"], res["peak_polyp_prob"], res["location"] = 0.0, 0.0, "Uninformative frame"
                    annotated, hud, hud_bg, saliency = rgb_arr.copy(), f"{timestamp:.1f}s | blur / wall-contact", (71, 85, 105), -1.0
                else:
                    cls_alpha = F.softplus(model.cls_model(tensor)) + 1.0
                    cur_probs = (cls_alpha / cls_alpha.sum(dim=-1, keepdim=True))[0].cpu().numpy()
                    ema_probs = cur_probs if ema_probs is None else (0.55 * cur_probs + 0.45 * ema_probs)
                    smooth_idx = int(np.argmax(ema_probs))
                    res["pred_label"], res["pred_conf"] = CLASS_NAMES[smooth_idx], float(ema_probs[smooth_idx])
                    annotated = res["blended"].copy()
                    hud, hud_bg = f"{timestamp:.1f}s | {res['pred_label']} ({res['pred_conf']*100:.0f}%)", (15, 23, 42)
                    saliency = res["peak_polyp_prob"] + (0.20 if "polyp" in res["pred_label"] else 0.0)
                cv2.rectangle(annotated, (0, 0), (224, 22), hud_bg, -1)
                cv2.putText(annotated, hud, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (248, 250, 252), 1)
                writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                frame_records.append((saliency, timestamp, tensor, res))
            frame_idx += 1
    cap.release()
    writer.release()
    if not frame_records:
        raise HTTPException(status_code=400, detail="Could not decode frames from video.")
    subprocess.run([ffmpeg_bin, "-y", "-i", raw_out, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", h264_out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    top_frames = sorted(sorted(frame_records, key=lambda x: x[0], reverse=True)[:4], key=lambda x: x[1])
    polyp_frames = sum(1 for r in frame_records if r[3]["polyp_area_pct"] > 0.5)
    timeline_lines, keyframes_out, keyframe_cache = [], [], []
    with torch.inference_mode():
        for idx, (_, ts, tensor, res) in enumerate(top_frames):
            cap_text = model.generate(tensor, max_new_tokens=64)[0].strip()
            timeline_lines.append(f"- [t={ts:.1f}s] {res['pred_label']} ({res['pred_conf']*100:.1f}%, EDL Unc: {res['cls_uncertainty']:.3f}, Mask: {res['polyp_area_pct']:.1f}%) | {cap_text}")
            keyframes_out.append({"index": idx, "timestamp": round(ts, 1), "label": res["pred_label"], "confidence": res["pred_conf"], "thumb": np_to_b64(res["blended"])})
            keyframe_cache.append({"timestamp": ts, "rgb": res["rgb"], "polyp_prob": res["polyp_prob"], "uncertainty_map": res["uncertainty_map"], "pred_label": res["pred_label"], "pred_conf": res["pred_conf"], "top3": res["top3"], "cls_uncertainty": res["cls_uncertainty"], "caption": cap_text})

    peak_rec = max(frame_records, key=lambda x: x[0])
    peak_res = peak_rec[3]
    peak_idx = next((i for i, kf in enumerate(keyframe_cache) if abs(kf["timestamp"] - peak_rec[1]) < 1e-4), 0)
    structured_context = (
        f"MODALITY: Colonoscopy Video ({len(frame_records)} sampled frames @ {sample_fps} FPS)\n"
        f"POLYP DETECTION RATE: {polyp_frames}/{len(frame_records)} frames\n"
        f"PEAK FRAME FINDING: {peak_res['pred_label']} ({peak_res['pred_conf']*100:.1f}%, EDL Unc: {peak_res['cls_uncertainty']:.3f})\n"
        f"KEYFRAME TIMELINE:\n" + "\n".join(timeline_lines)
    )
    SESSION_CACHE[session_id] = {"mode": "video", "keyframes": keyframe_cache, "active_idx": peak_idx, "structured_context": structured_context}
    return {
        "session_id": session_id, "mode": "video", "raw_image": np_to_b64(peak_res["rgb"]),
        "overlay_image": np_to_b64(peak_res["blended"]), "uncertainty_image": np_to_b64(peak_res["uncertainty_map"]),
        "mask_image": np_to_b64(peak_res["binary_mask_rgb"]), "video_url": f"/outputs/overlay_{session_id}.mp4",
        "keyframes": keyframes_out,
        "metrics": {
            "finding": peak_res["pred_label"], "confidence": peak_res["pred_conf"], "uncertainty": peak_res["cls_uncertainty"],
            "coverage": peak_res["polyp_area_pct"], "location": peak_res["location"], "top3": peak_res["top3"],
            "caption": keyframe_cache[peak_idx]["caption"], "structured_context": structured_context,
        },
    }

class ThresholdReq(BaseModel):
    session_id: str
    mask_thresh: float
    keyframe_idx: Optional[int] = None

@app.post("/api/threshold")
async def api_threshold(req: ThresholdReq):
    sess = SESSION_CACHE.get(req.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    if sess["mode"] == "video":
        if req.keyframe_idx is not None:
            sess["active_idx"] = req.keyframe_idx
        item = sess["keyframes"][sess.get("active_idx", 0)]
    else:
        item = sess
    blended, mask_512_rgb, area_pct, location = render_overlay_from_prob(item["rgb"], item["polyp_prob"], req.mask_thresh)
    return {
        "raw_image": np_to_b64(item["rgb"]), "overlay_image": np_to_b64(blended),
        "uncertainty_image": np_to_b64(item["uncertainty_map"]), "mask_image": np_to_b64(mask_512_rgb),
        "metrics": {"finding": item["pred_label"], "confidence": item["pred_conf"], "uncertainty": item["cls_uncertainty"], "coverage": area_pct, "location": location, "top3": item["top3"], "caption": item["caption"]},
    }

class ChatReq(BaseModel):
    message: str
    history: List[dict] = []
    structured_context: str = ""
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"

@app.post("/api/chat/stream")
async def api_chat_stream(req: ChatReq):
    tokenizer, llm = get_chat_llm(req.model_id)
    ctx = req.structured_context or "No scan analyzed yet. Answer general GI endoscopy questions."
    system_prompt = (
        "You are a senior gastroenterology AI clinical assistant. Interpret the structured outputs from "
        "GastroNet DINOv1 ViT-B/16, Hierarchical EDL classifier, ViTUNet segmenter, and Qwen2.5-0.5B captioner.\n\n"
        f"--- CURRENT CASE TELEMETRY ---\n{ctx}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    for turn in req.history[-8:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": str(turn["content"])})
    messages.append({"role": "user", "content": req.message})
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(DEVICE)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def _generate():
        with torch.inference_mode():
            amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if DEVICE.type == "cuda" else torch.no_grad()
            with amp_ctx:
                llm.generate(**inputs, streamer=streamer, max_new_tokens=512, temperature=0.3, top_p=0.9, do_sample=True)

    thread = threading.Thread(target=_generate)
    thread.start()
    def _event_stream():
        for chunk in streamer:
            yield json.dumps({"token": chunk}) + "\n"
        thread.join()
    return StreamingResponse(_event_stream(), media_type="application/x-ndjson", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

@app.post("/api/report")
async def api_generate_report(req: ChatReq):
    if not req.structured_context.strip():
        raise HTTPException(status_code=400, detail="Analyze a scan first.")
    tokenizer, llm = get_chat_llm(req.model_id)
    notes = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in req.history[-6:] if m.get("content")])
    instruction = (
        "Write a concise, minimalistic clinical summary (max 5 short bullet points) covering: "
        "1) Primary Finding & Location, 2) Lesion Coverage, 3) Uncertainty Level, and 4) Recommended Next Step. "
        "Keep it crisp and direct without filler headers.\n\n"
        f"TELEMETRY:\n{req.structured_context}\n\nNOTES:\n{notes}"
    )
    messages = [{"role": "system", "content": "You are an expert gastroenterology report writer."}, {"role": "user", "content": instruction}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if DEVICE.type == "cuda" else torch.no_grad()
        with amp_ctx:
            gen_ids = llm.generate(**inputs, max_new_tokens=600, temperature=0.25, do_sample=True)
    full_out = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    marker = next((c for c in ("VISUAL_PROMPT:", "VISUAL_GENERATION_PROMPT:") if c in full_out), None)
    if marker:
        rep_part, prm_part = full_out.split(marker, 1)
        detailed_report = rep_part.rstrip("#* \n").strip()
        lines = [ln.strip(" *#") for ln in prm_part.strip().splitlines() if ln.strip(" *#")]
        visual_prompt = lines[0] if lines else "sessile polyp on erythematous colonic mucosa"
    else:
        detailed_report, visual_prompt = full_out, "sessile polyp on erythematous colonic mucosa"
    return {"report": detailed_report, "visual_prompt": visual_prompt}

class PdfReq(BaseModel):
    overlay_image: Optional[str] = None
    uncertainty_image: Optional[str] = None
    structured_context: str = ""
    report: str = ""

def _get_font(size: int, bold: bool = False):
    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    )
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


@app.post("/api/export/pdf")
async def api_export_pdf(req: PdfReq):
    W, H = 1240, 1754  # A4 @ 150 DPI
    page = Image.new("RGB", (W, H), (248, 250, 252))
    draw = ImageDraw.Draw(page)

    f_title = _get_font(26, bold=True)
    f_sub = _get_font(15, bold=False)
    f_sec = _get_font(19, bold=True)
    f_lbl = _get_font(14, bold=True)
    f_val = _get_font(15, bold=False)
    f_body = _get_font(15, bold=False)
    f_small = _get_font(13, bold=False)

    # 1. Top Clinical Header Banner
    draw.rectangle([0, 0, W, 118], fill=(15, 23, 42))
    draw.rectangle([0, 112, W, 118], fill=(14, 165, 233))
    draw.rounded_rectangle([48, 24, 104, 80], radius=12, fill=(2, 132, 199))
    draw.text((62, 38), "EV", fill=(255, 255, 255), font=f_title)

    draw.text((124, 26), "ENDOVISION AI — GI ENDOSCOPY DIAGNOSTIC REPORT", fill=(248, 250, 252), font=f_title)
    draw.text(
        (124, 64),
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
        fill=(148, 163, 184),
        font=f_sub,
    )

    # Parse structured_context lines into clean key-value pairs
    label_Map = {
        "CAPTION": "CAPTION",
        "SEGMENTATION MASK COVERAGE": "MASK COVERAGE",
        "EDL EPISTEMIC UNCERTAINTY": "EDL UNCERTAINTY",
    }
    kv_pairs = []
    for raw_line in (req.structured_context or "").splitlines():
        if ":" in raw_line and not raw_line.strip().startswith("-"):
            k, v = raw_line.split(":", 1)
            k_clean = label_Map.get(k.strip(), k.strip())
            kv_pairs.append((k_clean, v.strip()))
        elif raw_line.strip():
            kv_pairs.append(("TIMELINE", raw_line.strip()))

    # 2. Side-by-Side Diagnostic Figures (400 x 400 each so narrative never clips)
    y = 138
    fig_size = 400
    ov_np, un_np = b64_to_np(req.overlay_image), b64_to_np(req.uncertainty_image)

    # Left Figure Card
    draw.rounded_rectangle([48, y, 604, y + fig_size + 52], radius=12, fill=(255, 255, 255), outline=(203, 213, 225), width=2)
    if ov_np is not None:
        page.paste(Image.fromarray(ov_np).resize((fig_size, fig_size)), (126, y + 12))
    draw.text((115, y + fig_size + 20), "Fig 1. Lesion Segmentation Overlay", fill=(15, 23, 42), font=f_lbl)

    # Right Figure Card
    draw.rounded_rectangle([636, y, 1192, y + fig_size + 52], radius=12, fill=(255, 255, 255), outline=(203, 213, 225), width=2)
    if un_np is not None:
        page.paste(Image.fromarray(un_np).resize((fig_size, fig_size)), (714, y + 12))
    draw.text((668, y + fig_size + 20), "Fig 2. Epistemic Uncertainty Map", fill=(15, 23, 42), font=f_lbl)

    y += fig_size + 72

    # 3. Stage 1 Structured Telemetry Table
    draw.text((48, y), "1. PERCEPTION TELEMETRY", fill=(2, 132, 199), font=f_sec)
    y += 28

    for idx, (k, v) in enumerate(kv_pairs):
        wrapped_v = textwrap.wrap(v, width=78) or ["—"]
        row_h = max(32, 12 + len(wrapped_v) * 20)
        bg = (255, 255, 255) if idx % 2 == 0 else (241, 245, 249)
        draw.rectangle([48, y, W - 48, y + row_h], fill=bg, outline=(226, 232, 240), width=1)
        draw.text((64, y + 7), k, fill=(51, 65, 85), font=f_lbl)
        for w_i, w_line in enumerate(wrapped_v):
            draw.text((370, y + 7 + w_i * 20), w_line, fill=(15, 23, 42), font=f_val)
        y += row_h

    y += 24

    # 4. Clinical Interpretation & Follow-Up Narrative (Qwen2.5-7B)
    draw.text((48, y), "2. CLINICAL SUMMARY & FOLLOW-UP", fill=(13, 148, 136), font=f_sec)
    y += 28

    raw_rep = req.report or "No clinical narrative generated. Click 'Generate Clinical Report' prior to export."
    clean_rep = re.sub(r"[*#`]+", "", raw_rep).strip()
    # Strip echoed instruction header if present
    clean_rep = chr(10).join(ln for ln in clean_rep.splitlines() if "DETAILED CLINICAL REPORT" not in ln.upper()).strip()

    box_bottom = H - 70
    draw.rounded_rectangle([48, y, W - 48, box_bottom], radius=12, fill=(255, 255, 255), outline=(203, 213, 225), width=2)

    ty = y + 16
    for para in clean_rep.splitlines():
        if not para.strip():
            ty += 8
            continue
        is_heading = para.strip().endswith(":") and len(para.strip()) < 45
        active_font = f_lbl if is_heading else f_body
        active_color = (15, 23, 42) if is_heading else (30, 41, 59)
        for w_line in textwrap.wrap(para.strip(), width=106):
            if ty < box_bottom - 24:
                draw.text((68, ty), w_line, fill=active_color, font=active_font)
                ty += 22

    # 5. Footer
    draw.line([(48, H - 52), (W - 48, H - 52)], fill=(203, 213, 225), width=1)
    draw.text(
        (48, H - 38),
        "EndoVision AI Clinical Decision Support  •  Evidential uncertainty > 0.45 warrants secondary endoscopic & histopathological correlation.",
        fill=(100, 116, 139),
        font=f_small,
    )

    pdf_name = f"EndoVision_Report_{uuid.uuid4().hex[:6]}.pdf"
    page.save(f"{OUTPUTS_DIR}/{pdf_name}", "PDF", resolution=150.0)
    return {"pdf_url": f"/outputs/{pdf_name}"}


class SynthImgReq(BaseModel):
    prompt: str
    mask_b64: Optional[str] = None
    guidance_scale: float = 7.5
    steps: int = 25

@app.post("/api/synthesize/image")
async def api_synthesize_image(req: SynthImgReq):
    global IMG_GEN_PIPE, VID_GEN_PIPE
    if VID_GEN_PIPE is not None:
        del VID_GEN_PIPE
        VID_GEN_PIPE = None
    offload_reasoning_models()
    mask_np = b64_to_np(req.mask_b64)
    if mask_np is not None and mask_np.max() < 15:
        mask_np = None
    clean_desc = " ".join(re.sub(r"[*#`>-]+", " ", req.prompt or "").split()[:35])
    full_prompt = f"Raw endoscopic camera photo inside human colon, wet glossy pink-red vascular colonic mucosa, {clean_desc}, soft natural mucosal folds, bright endoscope xenon light reflection"
    neg_prompt = "gray, greyscale, monochrome, stone, crater, ring border, cartoon, 3d render, text, watermark, blurry"
    if mask_np is not None:
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        if not isinstance(IMG_GEN_PIPE, StableDiffusionXLControlNetPipeline):
            IMG_GEN_PIPE = None
            gc.collect()
            torch.cuda.empty_cache()
            cn = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16, variant="fp16")
            IMG_GEN_PIPE = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn, torch_dtype=torch.float16, variant="fp16")
            IMG_GEN_PIPE.enable_model_cpu_offload()
        mask_gray = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY) if mask_np.ndim == 3 else mask_np
        edges = cv2.Canny(cv2.GaussianBlur(cv2.resize(mask_gray, (1024, 1024)), (31, 31), 0), 40, 120)
        out = IMG_GEN_PIPE(prompt=full_prompt, negative_prompt=neg_prompt, image=Image.fromarray(np.stack([edges]*3, axis=-1)), controlnet_conditioning_scale=0.38, num_inference_steps=int(req.steps), guidance_scale=float(req.guidance_scale)).images[0]
    else:
        from diffusers import StableDiffusionXLPipeline
        if not isinstance(IMG_GEN_PIPE, StableDiffusionXLPipeline):
            IMG_GEN_PIPE = None
            gc.collect()
            torch.cuda.empty_cache()
            IMG_GEN_PIPE = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16")
            IMG_GEN_PIPE.enable_model_cpu_offload()
        out = IMG_GEN_PIPE(prompt=full_prompt, negative_prompt=neg_prompt, num_inference_steps=int(req.steps), guidance_scale=float(req.guidance_scale)).images[0]
    return {"image_b64": np_to_b64(np.array(out.resize((512, 512))))}

class SynthVidReq(BaseModel):
    seed_image_b64: str
    motion_prompt: str
    num_frames: int = 49
    steps: int = 35

@app.post("/api/synthesize/video")
async def api_synthesize_video(req: SynthVidReq):
    global IMG_GEN_PIPE, VID_GEN_PIPE
    seed_np = b64_to_np(req.seed_image_b64)
    if seed_np is None:
        raise HTTPException(status_code=400, detail="Provide a starting endoscopic frame first.")
    if IMG_GEN_PIPE is not None:
        del IMG_GEN_PIPE
        IMG_GEN_PIPE = None
    offload_reasoning_models()
    from diffusers import CogVideoXImageToVideoPipeline
    from diffusers.utils import export_to_video
    if VID_GEN_PIPE is None:
        VID_GEN_PIPE = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
        VID_GEN_PIPE.enable_model_cpu_offload()
        VID_GEN_PIPE.vae.enable_slicing()
        VID_GEN_PIPE.vae.enable_tiling()
    clean_motion = " ".join(re.sub(r"[*#`>-]+", " ", req.motion_prompt or "").split()[:30])
    pil_seed = Image.fromarray(seed_np).convert("RGB").resize((720, 480))
    prompt = f"Subtle steady medical colonoscope camera view inside wet pink colon lumen, {clean_motion}, natural mucosal motion, realistic wet specular highlights"
    neg_prompt = "morphing, melting, warping, fast camera spin, cartoon, CGI, blurry, low quality"
    frames = VID_GEN_PIPE(prompt=prompt, negative_prompt=neg_prompt, image=pil_seed, num_videos_per_prompt=1, num_inference_steps=int(req.steps), num_frames=int(req.num_frames), guidance_scale=6.0, use_dynamic_cfg=True).frames[0]
    vid_name = f"synth_colon_{uuid.uuid4().hex[:6]}.mp4"
    export_to_video(frames, f"{OUTPUTS_DIR}/{vid_name}", fps=8)
    return {"video_url": f"/outputs/{vid_name}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
