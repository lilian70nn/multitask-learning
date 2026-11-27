# Deploying EndoVision AI on helena

Written for a coding agent working on **helena** (RTX 4090, 24 GB) with no other
context. Follow the steps in order. Every step ends with a check; do not continue
until it passes.

## 0. Rules for agents

1. **Ask the user before** running anything with `sudo`, installing system
   packages, editing `/etc`, changing the firewall, changing DNS, or doing anything
   that makes the service reachable from outside the machine. Show the exact
   command first. A yes for one command is not a yes for the next.
2. **Never print, log or commit the password.** Do not `cat`, `grep` or open
   `.env` in a way that shows `APP_PASSWORD`. Never put it on a command line, in a
   unit file, in the Caddyfile or in a chat message.
3. Never commit `.env`, model weights, `outputs/`, images or videos.
4. **Do not enable synthesis** (`ENABLE_SYNTHESIS=1`). SDXL and CogVideoX do not fit
   next to the other models on a 4090, and the web page does not use them.
5. Keep the app on `127.0.0.1`. Never set `HOST=0.0.0.0`.
6. This is a research demo, **not for clinical use**. Keep the banner. Do not upload
   data that can identify a patient.
7. helena may be shared. Do not kill other people's processes or GPU jobs.

## 1. What you are deploying

```
browser --HTTPS:443--> Caddy --HTTP--> 127.0.0.1:7860  python server.py (FastAPI)
                                        |- static/index.html (login required)
                                        |- vision + caption stack, fp16, on the GPU
                                        `- Qwen2.5-7B-Instruct, 4-bit NF4, on the GPU
```

One process. Analysis and login sessions are lost on restart, generated files go to
`<repo>/outputs/` (deleted after `OUTPUT_TTL_HOURS`). One GPU lock: analyse, chat
and report requests run one at a time; a request that waits more than 300 s gets
"The GPU is busy" (503). Models load on first use, so the first analysis and the
first chat after a start are slow.

Code: `server.py` (API, model loading), `web_auth.py` (login and request limits),
`static/` (pages and vendored JavaScript), `requirements_web.txt` (pinned versions),
`scripts/download_weights.py`, `scripts/smoke_test_webapp.py`, `scripts/parity_fp16.py`.

Memory target: **peak below 12 GB** of GPU memory.

Files in `deploy/helena/`: `.env.example`, `install.sh`, `endovision.service`
(systemd template), `Caddyfile.example`, this document.

## 2. Requirements

| Requirement | Check |
|---|---|
| NVIDIA driver 535 or newer | `nvidia-smi --query-gpu=driver_version,name,memory.total,memory.used --format=csv` |
| Python 3.10 to 3.12, or `uv` (3.10 is the tested version) | `python3 --version`; `uv --version` |
| CUDA 12 runtime | comes with the torch cu121 wheels; no system CUDA toolkit needed |
| ffmpeg | `ffmpeg -version` (the app itself uses the binary shipped by `imageio-ffmpeg`) |
| DejaVu font | `ls /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf` |
| about 30 GB free disk | `df -h . ~` |
| internet access to github.com, pypi.org, download.pytorch.org, huggingface.co | |

If something is missing, ask the user before running, for example:

```bash
sudo apt install ffmpeg fonts-dejavu-core python3-venv
```

Without `uv`, the matching venv package is needed too (for example
`python3.10-venv`); the installer says so if it is missing.

Do not install or upgrade the NVIDIA driver on your own.

## 3. Get the code

```bash
git clone -b duc/pretraining https://github.com/lilian70nn/multitask-learning.git
cd multitask-learning
```

Run all later commands from this directory (called `<repo>` below). The path must
not contain spaces. If the repository is already cloned, use section 10.

**Check:**

```bash
git branch --show-current          # duc/pretraining
ls server.py requirements_web.txt scripts/download_weights.py deploy/helena/install.sh
```

If files are missing, the branch has not been pushed completely. Tell the user; do
not rewrite the files yourself. If git asks for credentials, ask the user.

## 4. Install

```bash
bash deploy/helena/install.sh --check-only    # reports problems, changes nothing
bash deploy/helena/install.sh                 # venv, packages, weights, model cache, .env
```

The script is idempotent: after a failure, fix the cause and run it again. It never
uses sudo. It creates `<repo>/.venv`, installs the pinned `requirements_web.txt`
(including torch for CUDA 12.1), runs `scripts/download_weights.py` (about 5.5 GB), caches
`microsoft/swin-tiny-patch4-window7-224` and `Qwen/Qwen2.5-7B-Instruct` (about 15 GB)
and creates `.env` without overwriting an existing one. The download takes a while;
run it in the background or with a long timeout.

To put the weights or the Hugging Face cache on another disk, create `.env` first
(`install -m 600 deploy/helena/.env.example .env`), set `WEIGHTS_DIR` / `HF_HOME`
there as absolute paths, then run the installer.

**Check:** the script ends with `Done`, and:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

must print a `+cu121` version, `True` and the RTX 4090.

## 5. Configure `.env`

The app refuses to start without `APP_USER` and `APP_PASSWORD`.

Preferred: ask the user to edit the file in their own terminal
(`nano <repo>/.env`) and set `APP_USER` and `APP_PASSWORD` (long and random; wrap in
single quotes if it contains spaces, `#` or `$`).

If the user asks you to create the password, this writes a random one into `.env`
**without printing it** (only if the password is still empty). Set the username they
want; the user reads the password from the file themselves.

```bash
.venv/bin/python - <<'PY'
import re, secrets
path, username = ".env", "demo"          # <- username chosen by the user
text = open(path).read()
text = re.sub(r"(?m)^APP_USER=[ \t]*$", lambda m: "APP_USER=" + username, text)
text, n = re.subn(r"(?m)^APP_PASSWORD=[ \t]*$",
                  lambda m: "APP_PASSWORD=" + secrets.token_urlsafe(24), text)
open(path, "w").write(text)
print("password written" if n else "password already set; unchanged")
PY
chmod 600 .env
```

Leave the other values at their defaults on helena:

| Variable | Value | Meaning |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `7860` | local only; Caddy proxies to it |
| `COOKIE_SECURE` | `1` | cookie over HTTPS only (`0` only for plain-HTTP localhost or tunnel tests) |
| `ALLOW_NO_AUTH` | `0` | keep |
| `CHAT_MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` | chat and report model |
| `CHAT_QUANT` | `nf4` | 4-bit |
| `VISION_DTYPE` | `fp16` | half precision |
| `ENABLE_SYNTHESIS` | `0` | keep |
| `MAX_UPLOAD_MB` / `MAX_VIDEO_SECONDS` / `OUTPUT_TTL_HOURS` | `50` / `60` / `24` | limits |
| `SESSION_TTL_HOURS` | `12` | login lifetime |
| `SESSION_SECRET` | unset | cookie signing key; unset means a restart logs users out. If set: long, random, secret like the password |
| `WEIGHTS_DIR`, `HF_HOME`, `MULTITASK_REPO` | unset | only for non-default locations |
| `SIMULATE_VRAM_GB` | unset | testing on other GPUs only |

**Check** (shows no secrets):

```bash
grep -c '^APP_USER=.\+' .env         # 1
grep -c '^APP_PASSWORD=.\+' .env     # 1
stat -c %a .env                      # 600
git check-ignore .env                # prints .env
```

## 6. Smoke test (before enabling the service)

Check that the GPU has at least about 12 GB free (`nvidia-smi`). You need one
endoscopy image of your own (JPG or PNG; ask the user for it, or for a public frame
such as one from the HyperKvasir dataset). The LRZ test data is not on helena. Any
photo exercises the pipeline too, but then the predictions are meaningless. Keep
test images outside the repository.

```bash
.venv/bin/python scripts/smoke_test_webapp.py --image /path/to/frame.jpg
# or with several frames and your own short video:
.venv/bin/python scripts/smoke_test_webapp.py --images-dir /path/to/frames --video /path/to/clip.mp4
```

The script starts its own server on a free localhost port (reading `.env`), logs in,
tests health, image and video analysis, threshold, chat, report, PDF export, access
control and logout, then stops the server. Without `--video` it builds a short clip
from the images. The first run loads the models and takes longer. Instead of passing
`--image` or `--images-dir`, you can set the environment variable `TEST_IMAGES_DIR`
to a directory of test images; the precision check below reads it too. There is no
built-in default directory.

**Check:** it ends with `SMOKE TEST PASSED` and exit code 0, and `PEAK VRAM` is
below 12 GB. The printed health line must show `"vision_dtype": "fp16"`,
`"chat_quant": "nf4"`, `"synthesis_enabled": false` and `"device": "cuda"` for the
models. If not, read the printed server log tail and go to section 12. Do not enable
the service while this fails. (Exit code 2 means the test could not run, for example
a missing image or a server that did not start.)

Optional precision check, if the user has about 20 endoscopy images (nothing else
should be using much GPU memory):

```bash
.venv/bin/python scripts/parity_fp16.py --images-dir /path/to/images
```

It must end with `PARITY PASSED` (fp16 and fp32 agree).

## 7. systemd service (needs sudo: ask the user first)

From `<repo>`, as the user who ran the installer (the service must run as that user,
so that it finds the same model cache):

```bash
sed -e "s|__USER__|$(id -un)|g" -e "s|__REPO__|$PWD|g" -e "s|__VENV__|$PWD/.venv|g" \
    deploy/helena/endovision.service | sudo tee /etc/systemd/system/endovision.service
sudo systemctl daemon-reload
sudo systemctl enable --now endovision
```

If you are root or the user wants a dedicated service account, replace `__USER__`
accordingly and make sure that account can read the repository, `.env` and the model
cache. Do not run the service as root.

**Check:**

```bash
systemctl status endovision --no-pager                 # active (running)
curl -s http://127.0.0.1:7860/api/health               # JSON (wait if just started)
ss -ltn | grep 7860                                    # 127.0.0.1:7860 only
curl -s -o /dev/null -w '%{http_code}\n' -X POST -H 'Content-Type: application/json' \
     -d '{"session_id":"x","mask_thresh":0.5}' http://127.0.0.1:7860/api/threshold   # 401
.venv/bin/python scripts/smoke_test_webapp.py --url http://127.0.0.1:7860 --image /path/to/frame.jpg
```

Without sudo, the app can be run in a `tmux` session with
`.venv/bin/python server.py` for testing; it then does not restart on its own.

## 8. HTTPS with Caddy (exposes the demo: ask the user first)

Ask the user for the domain or subdomain. Its DNS record must point to helena, and
ports 80 and 443 must be reachable from the internet. Ask whether helena already
runs a web server on 80/443 (`ss -ltn | grep -E ':(80|443) '`); if so, do not
replace it, discuss it first.

1. Install Caddy v2 following https://caddyserver.com/docs/install.
2. Add the site block of `deploy/helena/Caddyfile.example` to `/etc/caddy/Caddyfile`
   with the real domain. Back up the existing file; keep other sites.
3. Validate and reload:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

**Check** (from outside helena if possible; replace the domain):

```bash
curl -sI https://demo.example.org/            # redirect (302/303/307) to /login
curl -sI http://demo.example.org/             # redirect to https
curl -s  https://demo.example.org/api/health  # JSON
curl -s -o /dev/null -w '%{http_code}\n' https://demo.example.org/static/index.html   # not 200
```

Then ask the user to log in with a browser, analyse an image and a short video, send
a chat message (the answer must appear word by word), generate a report and export
the PDF. To show the demo on their website, they link to `https://<domain>/`.

`/api/health` is public by design and shows GPU and model information only. To hide
it from the internet, add inside the site block: `respond /api/health 404`.

**No public address or domain:** do not open anything. The user can use an SSH
tunnel (`ssh -L 7860:127.0.0.1:7860 user@helena`, then `http://localhost:7860`) with
`COOKIE_SECURE=0` in `.env` and a service restart. Set it back to `1` before using
Caddy. Discuss any other tunnelling service with the user before installing it.

## 9. Firewall (needs sudo: ask the user first)

Only 80 and 443 (and the existing SSH access) should be reachable. Port 7860 must
never be opened. Look first:

```bash
sudo ufw status verbose      # or: sudo nft list ruleset / sudo firewall-cmd --list-all
ss -ltn                      # what is listening
```

For ufw. Allow SSH **before** enabling, use the real SSH port if it is not 22, and
ask whether other services on helena need ports, or enabling the firewall will cut
them off:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

**Check:** a new SSH session still works; from another machine
`curl -m 5 http://<helena-address>:7860/api/health` fails.

## 10. Update

```bash
cd <repo>
git status --short                 # local changes? ask the user before touching them
git rev-parse HEAD                 # note it, for rollback
git pull --ff-only
bash deploy/helena/install.sh      # new dependencies or weights; keeps .env
diff <(grep -oE '^#? ?[A-Z_]+=' deploy/helena/.env.example | tr -d '# ' | sort -u) \
     <(grep -oE '^#? ?[A-Z_]+=' .env | tr -d '# ' | sort -u)     # new settings? (names only)
sudo systemctl restart endovision  # ask first
.venv/bin/python scripts/smoke_test_webapp.py --url http://127.0.0.1:7860 --image /path/to/frame.jpg
```

If `deploy/helena/endovision.service` changed, repeat section 7. Rollback:
`git checkout <old commit>`, run the installer, restart; tell the user.

Changes to `.env` also need `sudo systemctl restart endovision`.

## 11. Logs and memory

```bash
journalctl -u endovision -n 200 --no-pager
journalctl -u endovision -f
journalctl -u caddy -n 100 --no-pager
```

(Reading the journal may need membership in the `systemd-journal` group or sudo.)

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
curl -s http://127.0.0.1:7860/api/health | python3 -m json.tool
```

| `/api/health` field | Meaning |
|---|---|
| `vram_used_gb`, `vram_reserved_gb` | memory allocated / reserved by PyTorch now |
| `vram_peak_gb`, `vram_peak_reserved_gb` | peaks since the service started; the 12 GB target and the smoke test use `vram_peak_gb` |
| `vram_total_gb`, `gpu` | the card |
| `vision_dtype`, `chat_model`, `chat_quant`, `synthesis_enabled` | active settings |
| `models.vision`, `models.chat` | what is loaded, precision and device (empty until first use) |

`nvidia-smi` shows more than `vram_peak_gb` (CUDA context and PyTorch's cache).

## 12. Troubleshooting

**The service does not start / restarts in a loop.** Read the journal. `Refusing to
start: APP_USER and APP_PASSWORD must be set` means `.env` is incomplete, not in
`<repo>`, or not readable by the service user. A `ValueError` naming a variable means
an invalid value in `.env`. Check `User=` and the paths in
`/etc/systemd/system/endovision.service`.

**`Missing model weights: ...`** in the log (analysis fails): run
`.venv/bin/python scripts/download_weights.py --check-only`, then without the flag to
fetch what is missing. Check `WEIGHTS_DIR`.

**bitsandbytes / CUDA errors** (`libbitsandbytes ... .so`, `CUDA Setup failed`,
`undefined symbol`, `torch.cuda.is_available()` is False):
check `nvidia-smi` and the driver version (535+; after a driver update a reboot may
be needed, ask the user). Check the wheels:
`.venv/bin/python -c "import torch,bitsandbytes; print(torch.__version__, torch.version.cuda, bitsandbytes.__version__)"`
must show `+cu121` and `12.1`. If a CPU or other CUDA build got installed, run the
installer again, or recreate the venv (remove `<repo>/.venv`, run the installer). Do
not upgrade single packages past the pins in `requirements_web.txt`
(`transformers==5.16.1` is deliberate: the checkpoints need 5.x names and later
releases need a newer torch). Make sure `LD_LIBRARY_PATH` does not point to another
CUDA installation.

**Out of memory** (`CUDA out of memory`): look at
`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`. Another
process may hold memory (also a second copy of this app, or a smoke test running
next to the service): ask the user before stopping anything. Confirm `CHAT_QUANT=nf4`,
`VISION_DTYPE=fp16`, `ENABLE_SYNTHESIS=0`, and that `SIMULATE_VRAM_GB` is unset. Then
restart the service.

**Port conflict** (`address already in use`): `ss -ltnp | grep 7860`. Stop your own
leftover `server.py`; for another service, set `PORT` in `.env` **and** in
`Environment=PORT=` of the unit file (the unit wins over `.env`) **and** in the
Caddyfile.

**Hugging Face download errors** (timeouts, 429, `LocalEntryNotFoundError`, 401):
re-run the installer (downloads resume). Check disk space and
`curl -sI https://huggingface.co`. For rate limits the user may set `HF_TOKEN` in
their shell (never write it into the repo or print it). Behind a proxy, set
`HTTPS_PROXY`. If the installer works but the service downloads again or fails, the
service uses another cache: set the same `HF_HOME` in `.env` and check `User=`.
`scripts/download_weights.py` reports missing weight files by name.

**Login page comes back after logging in:** `COOKIE_SECURE=1` while using plain
HTTP. Use HTTPS (Caddy), or `COOKIE_SECURE=0` for tunnel tests only.

**"Too many failed attempts" (429):** five wrong logins from one address within five
minutes. Wait five minutes (a service restart also clears it).

**"Cross-origin request rejected" (403):** the browser's `Origin` does not match the
`Host` the app receives. Keep Caddy's default (it passes `Host` through; no
`header_up Host`) and no `Referrer-Policy no-referrer` header in the proxy.

**Upload rejected (413):** larger than `MAX_UPLOAD_MB` or a video longer than
`MAX_VIDEO_SECONDS`. When raising the size, also raise `max_size` in the Caddyfile.
A 413 without a JSON message comes from Caddy.

**"The GPU is busy" (503):** another request held the GPU for over 300 s. A 503 about
GPU memory is the out-of-memory case above. A chat answer ending in "Generation
failed on the server" has its cause in the journal.

**502 from Caddy:** the app is down or still starting; check the service and the port.

**Chat answer arrives all at once or times out:** keep `flush_interval -1` and the
long timeouts from `Caddyfile.example`; check for another proxy or CDN in front.

**Video analysis fails / video does not play:** check that
`.venv/bin/python -c "import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())"` works
and that the journal shows no decoding error. Try an H.264 MP4.

**PDF text tiny or unstyled:** install `fonts-dejavu-core` (ask first).

**"Session not found":** the service restarted; upload the file again.

**Disk fills up / `Disk quota exceeded`:** check
`du -sh .venv outputs model_weights ~/.cache/huggingface ~/.cache/uv ~/.cache/pip`,
the user's quota (`df` does not show quotas) and `OUTPUT_TTL_HOURS`. Ask the user
before deleting caches.

## Measured on LRZ (2026-11-27), before helena

One A100 with PyTorch memory capped at 24 GB (`SIMULATE_VRAM_GB=24`). Not yet run on an RTX 4090.

* fp16 vs fp32 on 20 HyperKvasir images, both segmentation domains: top-1 class 20/20 equal,
  largest class-probability difference 0.00007, minimum mask Dice 0.9974, captions identical for all 20
  images in both domains.
* Smoke test: 14/14 passed. Peak memory 7.0 GB allocated, 7.1 GB reserved by PyTorch
  (about 7.8 GB in `nvidia-smi` over the whole test job).
* The first chat request took about 52 s because the 7B model is loaded on first use (from a network
  file system on LRZ; a local disk should be faster).

## Known behaviour (from the original models and app, not from the deployment changes)

* The shown classification confidence is nearly constant (often 0.60 to 0.68) and the uncertainty
  near 0.39: in the hierarchical evidential classifier only the winning class receives evidence.
  Do not read the number as a calibrated probability. The same values come out of the original code.
* The chat model's clinical summary can state wrong numbers (seen once: the class confidence reported
  as the "detection rate"). Treat all generated text as unverified. Research demo, not for clinical use.

## 13. Security summary

- The app listens only on `127.0.0.1`; the unit file enforces it.
- Caddy provides HTTPS; the firewall opens only 80/443 (plus existing SSH).
- All pages, APIs, `/static` and `/outputs` need a login; only `/login`, `/logout`
  and `/api/health` do not. The smoke test checks this.
- The password lives only in `<repo>/.env` (mode 600), never in git, logs or chat.
  To change it, the user edits `.env` and the service is restarted.
- There is one shared account. Use a long random password; the login rate limit
  (5 failures per 5 minutes per address) is only a second line of defence.
- Uploads are limited in size and length; generated files are removed after
  `OUTPUT_TTL_HOURS`.
- Synthesis endpoints stay disabled.
- Keep the system and Caddy updated (user's decision and timing).

To take the demo offline (ask first): `sudo systemctl disable --now endovision`.
