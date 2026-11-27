# EndoVision AI

GI endoscopy research demo (not for clinical use): FastAPI `server.py` + React page
`static/index.html`.

**To install, deploy, update or debug the web app (target machine: helena, RTX 4090),
read [`deploy/helena/DEPLOY_HELENA.md`](deploy/helena/DEPLOY_HELENA.md) first and
follow it.**

Always:

- Ask the user before running `sudo`, changing the firewall or `/etc`, or exposing
  the service.
- Never print or commit the password or `.env`; never commit data, weights or outputs.
- Keep the app on `127.0.0.1` and keep synthesis disabled (`ENABLE_SYNTHESIS=0`).
- After a change, run `scripts/smoke_test_webapp.py` (GPU needed).
