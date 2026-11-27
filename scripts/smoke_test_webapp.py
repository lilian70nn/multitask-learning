#!/usr/bin/env python3
"""End-to-end smoke test of the EndoVision AI web app.

By default the script starts `python server.py` on a free localhost port, logs in,
runs login / health / analyze (image and video) / threshold / chat / report / PDF
export, checks the responses, prints peak VRAM and stops the server again.
With --url it tests a server that is already running instead.

Examples (from the repo root):

  .venv/bin/python scripts/smoke_test_webapp.py --image /path/to/frame.jpg
  .venv/bin/python scripts/smoke_test_webapp.py --images-dir /path/to/frames --video clip.mp4
  .venv/bin/python scripts/smoke_test_webapp.py --url http://127.0.0.1:7860 --image frame.jpg

Without --image and --images-dir, the directory is taken from the environment
variable TEST_IMAGES_DIR.

Without --video a short clip is generated from the given images in a temporary
directory. Credentials come from APP_USER / APP_PASSWORD (environment, else the
repo's .env). When the script starts the server itself and none are set, it uses
random throw-away credentials. Passwords are never printed.

Exit code: 0 = all checks passed, 1 = a check failed, 2 = could not run.
Only the standard library is needed, plus Pillow and imageio-ffmpeg.
"""
import argparse
import base64
import http.client
import io
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from urllib.parse import urlencode, urlsplit

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGES_DIR = os.environ.get("TEST_IMAGES_DIR", "")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
BOGUS_MODEL_ID = "smoke-test/this-model-id-must-be-ignored"
METRIC_KEYS = ["finding", "confidence", "uncertainty", "coverage", "location", "top3", "caption"]


class CheckFailed(Exception):
    pass


def die(message):
    """Stop because the test cannot run (exit code 2)."""
    print(message, file=sys.stderr)
    sys.exit(2)


def require(condition, message):
    if not condition:
        raise CheckFailed(message)


# --------------------------------------------------------------------------- HTTP
class Client:
    """Tiny HTTP client: keeps cookies and never follows redirects."""

    def __init__(self, base_url, timeout):
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"Invalid URL: {base_url}")
        self.scheme, self.host, self.port = parts.scheme, parts.hostname, parts.port
        self.prefix = parts.path.rstrip("/")
        self.timeout = timeout
        self.cookies = {}

    def open(self, method, path, body=None, headers=None, auth=True, timeout=None):
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        conn = cls(self.host, self.port, timeout=timeout or self.timeout)
        headers = dict(headers or {})
        if auth and self.cookies:
            headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in self.cookies.items())
        conn.request(method, self.prefix + path, body=body, headers=headers)
        return conn, conn.getresponse()

    def request(self, method, path, **kwargs):
        conn, resp = self.open(method, path, **kwargs)
        try:
            data = resp.read()
            if kwargs.get("auth", True):
                self.store_cookies(resp.headers)
            return resp.status, resp.headers, data
        finally:
            conn.close()

    def store_cookies(self, response_headers):
        for header in response_headers.get_all("Set-Cookie") or []:
            first = header.split(";", 1)[0]
            if "=" not in first:
                continue
            name, value = first.split("=", 1)
            expired = "max-age=0" in header.lower() or "01 jan 1970" in header.lower()
            if expired or value.strip('"') in ("", "null"):
                self.cookies.pop(name.strip(), None)
            else:
                self.cookies[name.strip()] = value

    def post_json(self, path, payload, **kwargs):
        return self.request("POST", path, body=json.dumps(payload).encode(),
                            headers={"Content-Type": "application/json"}, **kwargs)

    def post_form(self, path, fields, **kwargs):
        return self.request("POST", path, body=urlencode(fields).encode(),
                            headers={"Content-Type": "application/x-www-form-urlencoded"}, **kwargs)

    def post_file(self, path, fields, file_path, content_type, **kwargs):
        boundary = "----smoke" + uuid.uuid4().hex
        body = io.BytesIO()
        for name, value in fields.items():
            body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                       f"{value}\r\n".encode())
        # Neutral upload name: local file names are not needed by the server.
        upload_name = "upload" + os.path.splitext(file_path)[1].lower()
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                   f"filename=\"{upload_name}\"\r\nContent-Type: {content_type}\r\n\r\n".encode())
        with open(file_path, "rb") as handle:
            body.write(handle.read())
        body.write(f"\r\n--{boundary}--\r\n".encode())
        return self.request("POST", path, body=body.getvalue(),
                            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                            **kwargs)


def expect_json(status, data, what):
    require(status == 200, f"{what}: expected HTTP 200, got {status}: {data[:400]!r}")
    try:
        return json.loads(data)
    except ValueError:
        raise CheckFailed(f"{what}: response is not JSON: {data[:200]!r}")


def is_login_redirect(status, headers):
    return status in (302, 303, 307) and "/login" in (headers.get("Location") or "")


def check_png_data_url(value, field):
    from PIL import Image
    require(isinstance(value, str) and value.startswith("data:image/"),
            f"{field} is not an image data URL")
    image = Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1])))
    image.load()
    require(min(image.size) >= 32, f"{field} has an implausible size {image.size}")
    return image.size


def check_metrics(metrics, extra=()):
    require(isinstance(metrics, dict), "metrics is not an object")
    missing = [k for k in METRIC_KEYS + list(extra) if k not in metrics]
    require(not missing, f"metrics lacks {missing}")
    require(0.0 <= float(metrics["confidence"]) <= 1.0, f"confidence out of range: {metrics['confidence']}")
    require(0.0 <= float(metrics["uncertainty"]) <= 1.0001, f"uncertainty out of range: {metrics['uncertainty']}")
    require(0.0 <= float(metrics["coverage"]) <= 100.0, f"coverage out of range: {metrics['coverage']}")
    require(len(metrics["top3"]) == 3 and all("label" in t and "prob" in t for t in metrics["top3"]),
            "top3 must hold three {label, prob} entries")


# ------------------------------------------------------------------------- inputs
def find_images(args):
    if args.image:
        for path in args.image:
            if not os.path.isfile(path):
                die(f"ERROR: image not found: {path}")
        images = [os.path.abspath(p) for p in args.image]
        if not args.images_dir:
            return images
    else:
        images = []
    directory = args.images_dir or DEFAULT_IMAGES_DIR
    if directory:
        if not os.path.isdir(directory):
            source = "--images-dir" if args.images_dir else "TEST_IMAGES_DIR"
            die(f"ERROR: {source} is not a directory: {directory}")
        images += [os.path.join(directory, n) for n in sorted(os.listdir(directory))
                   if n.lower().endswith(IMAGE_EXTS)][:8]
    if not images:
        die("ERROR: no test image: pass --images-dir/--image or set TEST_IMAGES_DIR (endoscopy frames).")
    return images


def make_video(images, out_path, fps=10, frames_per_image=10, size=448):
    """Short H.264 clip showing each image for one second."""
    import imageio_ffmpeg
    from PIL import Image
    chosen = (images * 6)[:6] if len(images) < 6 else images[:6]
    writer = imageio_ffmpeg.write_frames(out_path, (size, size), fps=fps, codec="libx264",
                                         pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                         macro_block_size=16)
    writer.send(None)
    try:
        for path in chosen:
            frame = Image.open(path).convert("RGB").resize((size, size)).tobytes()
            for _ in range(frames_per_image):
                writer.send(frame)
    finally:
        writer.close()
    require(os.path.getsize(out_path) > 0, "generated video is empty")
    return out_path


# ------------------------------------------------------------------------- server
def read_dotenv(path):
    values = {}
    if not os.path.isfile(path):
        return values
    try:
        from dotenv import dotenv_values
        return {k: v for k, v in dotenv_values(path).items() if v is not None}
    except ImportError:
        pass
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            values[key.strip()] = value
    return values


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def log_tail(path, lines=40):
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-lines:])
    except OSError:
        return "(no server log)"


def start_server(args, user, password, port, log_path):
    server_py = os.path.join(args.repo, "server.py")
    if not os.path.isfile(server_py):
        die(f"ERROR: {server_py} not found (use --repo).")
    env = dict(os.environ)
    env.update({"HOST": "127.0.0.1", "PORT": str(port), "COOKIE_SECURE": "0",
                "APP_USER": user, "APP_PASSWORD": password, "ALLOW_NO_AUTH": "0",
                "PYTHONUNBUFFERED": "1"})
    log = open(log_path, "w")
    proc = subprocess.Popen([args.python, server_py], cwd=args.repo, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    return proc


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def wait_until_ready(client, proc, timeout, log_path):
    deadline = time.time() + timeout
    last_error = "no answer"
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            die(f"ERROR: server exited with code {proc.returncode} during startup.\n"
                     f"--- server log tail ({log_path}) ---\n{log_tail(log_path)}")
        try:
            status, _, _ = client.request("GET", "/api/health", auth=False, timeout=10)
            if status == 200:
                return
            last_error = f"HTTP {status}"
        except (OSError, http.client.HTTPException) as exc:
            last_error = str(exc)
        time.sleep(1)
    tail = f"\n--- server log tail ---\n{log_tail(log_path)}" if proc is not None else ""
    die(f"ERROR: server not ready after {timeout}s ({last_error}).{tail}")


# -------------------------------------------------------------------------- tests
class Runner:
    def __init__(self):
        self.failed, self.passed, self.skipped = [], 0, []

    def step(self, name, func, needs=True):
        if not needs:
            print(f"[SKIP] {name} (an earlier step failed)")
            self.skipped.append(name)
            return None
        start = time.time()
        try:
            result = func()
        except CheckFailed as exc:
            print(f"[FAIL] {name} ({time.time() - start:.1f}s): {exc}")
            self.failed.append(name)
            return None
        except Exception as exc:  # keep going, report at the end
            print(f"[FAIL] {name} ({time.time() - start:.1f}s): {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self.failed.append(name)
            return None
        self.passed += 1
        note = result.get("note", "") if isinstance(result, dict) else ""
        print(f"[ OK ] {name} ({time.time() - start:.1f}s) {note}".rstrip())
        return result if result is not None else {}


def get_health(client):
    status, _, data = client.request("GET", "/api/health", auth=False)
    health = expect_json(status, data, "health")
    for key in ("gpu", "vram_used_gb", "vram_total_gb"):
        require(key in health, f"health lacks '{key}': {sorted(health)}")
    return health


def peak_vram(health):
    """Peak memory allocated by PyTorch since the server started."""
    value = health.get("vram_peak_gb")
    return float(value) if isinstance(value, (int, float)) else None


def run_tests(client, args, user, password, images, video):
    runner = Runner()

    def health_step():
        health = get_health(client)
        require(peak_vram(health) is not None, f"health reports no peak VRAM: {sorted(health)}")
        require(not {"user", "username", "app_user"} & set(health), "health must not return user data")
        return {"note": f"gpu={health['gpu']} total={health['vram_total_gb']} GB", "health": health}

    runner.step("health without login", health_step)

    def unauth_step():
        status, _, _ = client.post_json("/api/threshold", {"session_id": "x", "mask_thresh": 0.5}, auth=False)
        require(status == 401, f"POST /api/threshold without login: expected 401, got {status}")
        status, _, _ = client.post_json("/api/chat/stream", {"message": "hi"}, auth=False)
        require(status == 401, f"POST /api/chat/stream without login: expected 401, got {status}")
        status, headers, _ = client.request("GET", "/", auth=False)
        require(is_login_redirect(status, headers), f"GET / without login: expected redirect to /login, got {status}")
        for path in ("/static/index.html", "/outputs/does-not-exist.pdf"):
            status, headers, _ = client.request("GET", path, auth=False)
            require(status == 401 or is_login_redirect(status, headers),
                    f"GET {path} without login: expected 401 or redirect, got {status}")
        status, _, data = client.request("GET", "/login", auth=False)
        require(status == 200 and b"password" in data.lower(), f"GET /login: no login form (HTTP {status})")

    if not args.no_auth:
        runner.step("access blocked without login", unauth_step)

    def bad_login_step():
        status, headers, _ = client.post_form("/login", {"username": user, "password": password + "-wrong"}, auth=False)
        require(not (status == 303 and headers.get("Set-Cookie") and "/login" not in (headers.get("Location") or "")),
                "a wrong password was accepted")
        probe = Client(args.base_url, client.timeout)
        probe.store_cookies(headers)
        status, _, _ = probe.post_json("/api/threshold", {"session_id": "x", "mask_thresh": 0.5})
        require(status == 401, f"cookie from a failed login grants access (HTTP {status})")

    def login_step():
        status, headers, data = client.post_form("/login", {"username": user, "password": password})
        require(status == 303, f"POST /login: expected 303, got {status}: {data[:200]!r}")
        require("/login" not in (headers.get("Location") or ""), "login redirected back to /login (wrong credentials?)")
        require(client.cookies, "login set no session cookie")
        status, _, page = client.request("GET", "/")
        require(status == 200 and b'id="root"' in page, f"GET / after login failed (HTTP {status})")

    logged_in = {}
    if not args.no_auth:
        runner.step("wrong password rejected", bad_login_step)
        logged_in = runner.step("login", login_step)

    def analyze(path, content_type):
        status, _, data = client.post_file(
            "/api/analyze", {"seg_domain": args.seg_domain, "mask_thresh": 0.5, "sample_fps": 2},
            path, content_type)
        body = expect_json(status, data, "analyze")
        require(body.get("session_id"), "no session_id")
        for field in ("raw_image", "overlay_image", "uncertainty_image", "mask_image"):
            check_png_data_url(body.get(field), field)
        check_metrics(body.get("metrics"), extra=["structured_context"])
        require(body["metrics"]["structured_context"].strip(), "empty structured_context")
        return body

    def image_step():
        ext = os.path.splitext(images[0])[1].lower()
        body = analyze(images[0], "image/png" if ext == ".png" else "image/jpeg")
        require(body["mode"] == "image", f"mode is {body['mode']!r}, expected 'image'")
        require(body.get("video_url") is None and body.get("keyframes") == [], "image result has video fields set")
        m = body["metrics"]
        return {"body": body, "note": f"{m['finding']} conf={m['confidence']:.3f} unc={m['uncertainty']:.3f} "
                                      f"mask={m['coverage']:.2f}%\n       caption: {m['caption']}"}

    image = runner.step("analyze image", image_step, needs=logged_in is not None)

    def threshold(session_id, thresh, **extra):
        status, _, data = client.post_json("/api/threshold",
                                           {"session_id": session_id, "mask_thresh": thresh, **extra})
        body = expect_json(status, data, "threshold")
        for field in ("raw_image", "overlay_image", "uncertainty_image", "mask_image"):
            check_png_data_url(body.get(field), field)
        check_metrics(body.get("metrics"))
        return body

    def threshold_image_step():
        sid = image["body"]["session_id"]
        low = threshold(sid, 0.1)["metrics"]["coverage"]
        high = threshold(sid, 0.9)["metrics"]["coverage"]
        require(low >= high, f"coverage should not grow with the threshold ({low} at 0.1, {high} at 0.9)")
        status, _, _ = client.post_json("/api/threshold", {"session_id": "no-such-session", "mask_thresh": 0.5})
        require(status == 404, f"unknown session: expected 404, got {status}")
        return {"note": f"coverage {low:.2f}% @0.1, {high:.2f}% @0.9"}

    runner.step("threshold (image)", threshold_image_step, needs=image is not None)

    def video_step():
        body = analyze(video, "video/mp4")
        require(body["mode"] == "video", f"mode is {body['mode']!r}, expected 'video'")
        frames = body.get("keyframes")
        require(isinstance(frames, list) and 1 <= len(frames) <= 4, "expected 1 to 4 keyframes")
        for frame in frames:
            missing = [k for k in ("index", "timestamp", "label", "confidence", "thumb") if k not in frame]
            require(not missing, f"keyframe lacks {missing}")
            check_png_data_url(frame["thumb"], "keyframe thumb")
        url = body.get("video_url")
        require(isinstance(url, str) and url.startswith("/outputs/"), f"bad video_url: {url!r}")
        status, _, data = client.request("GET", url)
        require(status == 200, f"GET {url}: HTTP {status} (overlay video missing; check ffmpeg)")
        require(len(data) > 1000 and b"ftyp" in data[:64], f"overlay video is not a valid MP4 ({len(data)} bytes)")
        if not args.no_auth:
            status, headers, _ = client.request("GET", url, auth=False)
            require(status == 401 or is_login_redirect(status, headers),
                    f"overlay video readable without login (HTTP {status})")
        return {"body": body, "note": f"{len(frames)} keyframes, overlay {len(data) // 1024} KiB"}

    vid = runner.step("analyze video", video_step, needs=logged_in is not None)

    def threshold_video_step():
        last = len(vid["body"]["keyframes"]) - 1
        threshold(vid["body"]["session_id"], 0.4, keyframe_idx=last)
        threshold(vid["body"]["session_id"], 0.6)

    runner.step("threshold (video keyframe)", threshold_video_step, needs=vid is not None)

    source = vid or image
    context = source["body"]["metrics"]["structured_context"] if source else ""

    def chat_step():
        payload = {"message": "In one sentence: what is the primary finding?", "history": [],
                   "structured_context": context, "model_id": BOGUS_MODEL_ID}
        start = time.time()
        conn, resp = client.open("POST", "/api/chat/stream", body=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
        try:
            if resp.status != 200:  # only then consume the body; success is read as a stream
                raise CheckFailed(f"expected HTTP 200, got {resp.status}: {resp.read()[:300]!r}")
            require("ndjson" in (resp.headers.get("Content-Type") or ""), "response is not NDJSON")
            reply, chunks, first = "", 0, None
            while True:
                line = resp.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                item = json.loads(line)
                require("token" in item, f"stream line without 'token': {item}")
                first = first or time.time() - start
                reply += item["token"]
                chunks += 1
        finally:
            conn.close()
        require(reply.strip(), "empty reply (see the server log)")
        require("[Generation failed on the server" not in reply, "chat generation failed (see the server log)")
        return {"reply": reply, "note": f"{chunks} chunks, first after {first:.1f}s, {len(reply)} chars"}

    chat = runner.step("chat stream (bogus model_id ignored)", chat_step, needs=bool(context))

    def report_step():
        history = [{"role": "assistant", "content": chat["reply"]}] if chat else []
        status, _, data = client.post_json("/api/report", {
            "message": "", "history": history, "structured_context": context, "model_id": BOGUS_MODEL_ID})
        body = expect_json(status, data, "report")
        require(str(body.get("report", "")).strip(), "empty report")
        require("visual_prompt" in body, "report response lacks 'visual_prompt'")
        status, _, _ = client.post_json("/api/report", {"message": "", "structured_context": ""})
        require(status == 400, f"report without analysis: expected 400, got {status}")
        return {"report": body["report"], "note": f"{len(body['report'])} chars"}

    report = runner.step("report", report_step, needs=bool(context))

    def pdf_step():
        body = source["body"]
        status, _, data = client.post_json("/api/export/pdf", {
            "overlay_image": body["overlay_image"], "uncertainty_image": body["uncertainty_image"],
            "structured_context": context, "report": report["report"] if report else ""})
        url = expect_json(status, data, "export/pdf").get("pdf_url")
        require(isinstance(url, str) and url.startswith("/outputs/") and url.endswith(".pdf"), f"bad pdf_url: {url!r}")
        status, _, pdf = client.request("GET", url)
        require(status == 200, f"GET {url}: HTTP {status}")
        require(pdf.startswith(b"%PDF") and len(pdf) > 5000, f"not a valid PDF ({len(pdf)} bytes)")
        if args.save_pdf:
            with open(args.save_pdf, "wb") as handle:
                handle.write(pdf)
        if not args.no_auth:
            status, headers, _ = client.request("GET", url, auth=False)
            require(status == 401 or is_login_redirect(status, headers), f"PDF readable without login (HTTP {status})")
        return {"note": f"{len(pdf) // 1024} KiB"}

    runner.step("export PDF", pdf_step, needs=source is not None)

    def synthesis_step():
        for path in ("/api/synthesize/image", "/api/synthesize/video"):
            # An empty body cannot start a generation even if the route exists.
            status, _, _ = client.post_json(path, {})
            require(status == 404, f"POST {path}: expected 404 (synthesis disabled), got {status}")

    if not args.expect_synthesis:
        runner.step("synthesis routes disabled", synthesis_step, needs=logged_in is not None)

    def logout_step():
        status, headers, _ = client.request("GET", "/logout")
        if status == 405:
            status, headers, _ = client.request("POST", "/logout")
        require(status in (200, 302, 303, 307), f"/logout: HTTP {status}")
        require(not client.cookies, "logout did not clear the session cookie")
        status, _, _ = client.post_json("/api/threshold", {"session_id": "x", "mask_thresh": 0.5})
        require(status == 401, f"API still reachable after logout (HTTP {status})")

    if not args.no_auth:
        runner.step("logout", logout_step, needs=logged_in is not None)

    def vram_step():
        health = get_health(client)
        print("       health: " + json.dumps(health))
        peak = peak_vram(health)
        require(peak is not None, "health reports no peak VRAM")
        models = health.get("models") or {}
        if image or vid:
            require(models.get("vision"), "health lists no loaded vision model")
        if chat or report:
            require(models.get("chat"), "health lists no loaded chat model")
        total = float(health.get("vram_total_gb") or 0)
        if total <= 0:
            return {"note": "no GPU reported; VRAM limit not checked"}
        reserved = health.get("vram_peak_reserved_gb")
        extra = "" if reserved is None else f"; peak reserved by PyTorch {reserved} GB"
        print(f"\nPEAK VRAM: {peak:.2f} GB allocated of {total:.1f} GB "
              f"(current {health['vram_used_gb']} GB{extra})\n")
        if args.max_vram_gb > 0:
            require(peak <= args.max_vram_gb, f"peak VRAM {peak:.2f} GB exceeds {args.max_vram_gb} GB")
        return {"note": f"limit {args.max_vram_gb} GB"}

    runner.step("VRAM after the full sequence", vram_step)
    return runner


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", action="append", help="test image (repeatable; the first one is analyzed)")
    p.add_argument("--images-dir", help="directory with endoscopy images, first 8 by name are used "
                                        "(without --image, default: $TEST_IMAGES_DIR)")
    p.add_argument("--video", help="test video; generated from the images if omitted")
    p.add_argument("--seg-domain", default="colonbench", choices=["colonbench", "polyp"])
    p.add_argument("--url", help="test a running server instead of starting one, e.g. http://127.0.0.1:7860")
    p.add_argument("--repo", default=REPO_DIR, help="repo root containing server.py (default: %(default)s)")
    p.add_argument("--python", default=sys.executable, help="interpreter used to start server.py")
    p.add_argument("--port", type=int, default=0, help="port for the started server (default: a free one)")
    p.add_argument("--server-log", help="log file of the started server (default: a new temp file, kept)")
    p.add_argument("--startup-timeout", type=int, default=900)
    p.add_argument("--timeout", type=int, default=900, help="per-request timeout in seconds (first calls load models)")
    p.add_argument("--max-vram-gb", type=float, default=12.0, help="fail above this peak VRAM; 0 disables")
    p.add_argument("--expect-synthesis", action="store_true", help="do not require /api/synthesize/* to be 404")
    p.add_argument("--no-auth", action="store_true", help="only with --url for a server run with ALLOW_NO_AUTH=1")
    p.add_argument("--save-pdf", help="also save the exported PDF to this path")
    return p


def main():
    args = build_parser().parse_args()
    args.repo = os.path.abspath(args.repo)
    if args.no_auth and not args.url:
        die("ERROR: --no-auth is only meaningful with --url.")
    images = find_images(args)
    if args.video and not os.path.isfile(args.video):
        die(f"ERROR: video not found: {args.video}")

    dotenv = read_dotenv(os.path.join(args.repo, ".env"))
    user = os.environ.get("APP_USER") or dotenv.get("APP_USER") or ""
    password = os.environ.get("APP_PASSWORD") or dotenv.get("APP_PASSWORD") or ""
    if not (user and password):
        if args.url and not args.no_auth:
            die("ERROR: set APP_USER and APP_PASSWORD (environment or .env) to test a running server.")
        user, password = user or "smoketest", password or secrets.token_urlsafe(24)

    with tempfile.TemporaryDirectory(prefix="endovision_smoke_") as tmp:
        video = os.path.abspath(args.video) if args.video else make_video(images, os.path.join(tmp, "clip.mp4"))
        proc, log_path = None, None
        if args.url:
            args.base_url = args.url
        else:
            port = args.port or free_port()
            args.base_url = f"http://127.0.0.1:{port}"
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    die(f"ERROR: port {port} is already in use (choose --port or use --url).")
            log_path = args.server_log
            if not log_path:
                handle, log_path = tempfile.mkstemp(prefix="endovision_smoke_server_", suffix=".log")
                os.close(handle)
        client = Client(args.base_url, args.timeout)
        try:
            if not args.url:
                proc = start_server(args, user, password, port, log_path)
                print(f"Started server.py (pid {proc.pid}) at {args.base_url}; log: {log_path}")
            # SIGTERM should also run the cleanup below.
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
            wait_until_ready(client, proc, args.startup_timeout, log_path)
            print(f"Testing {args.base_url} with image {images[0]}")
            runner = run_tests(client, args, user, password, images, video)
        finally:
            stop_server(proc)

    print(f"\n{runner.passed} passed, {len(runner.failed)} failed, {len(runner.skipped)} skipped")
    if runner.failed:
        print("Failed: " + ", ".join(runner.failed))
        if log_path:
            print(f"--- server log tail ({log_path}) ---\n{log_tail(log_path)}")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
