"""Session login and basic request hardening for the EndoVision web app.

A pure ASGI middleware covers every route (pages, /api, /static, /outputs).
Sessions are signed, expiring cookies (itsdangerous).
"""
import hashlib
import hmac
import secrets
import time
from html import escape
from urllib.parse import urlsplit

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from itsdangerous import BadSignature, URLSafeTimedSerializer
from starlette.datastructures import Headers, MutableHeaders
from starlette.requests import HTTPConnection

COOKIE_NAME = "endovision_session"
PUBLIC_PATHS = {"/login", "/logout", "/api/health"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

# Failed-login rate limit, per client address.
MAX_FAILURES = 5
FAILURE_WINDOW_S = 300


class AuthManager:
    def __init__(self, user: str, password: str, allow_no_auth: bool, cookie_secure: bool,
                 max_age_s: int, secret: str = ""):
        self.user, self.password = user, password
        self.allow_no_auth = allow_no_auth
        self.cookie_secure = cookie_secure
        self.max_age_s = max_age_s
        # Without SESSION_SECRET a random key is used, so a restart logs everybody out.
        self._secret = secret or secrets.token_hex(32)
        self._serializer = URLSafeTimedSerializer(self._secret, salt="endovision-session")
        self._revoked = set()
        self._failures = {}

    @property
    def enabled(self) -> bool:
        return bool(self.user and self.password)

    @property
    def open_access(self) -> bool:
        return not self.enabled and self.allow_no_auth

    def check_config(self) -> None:
        if not self.enabled and not self.allow_no_auth:
            raise RuntimeError(
                "APP_USER and APP_PASSWORD must be set (environment or .env). "
                "Set ALLOW_NO_AUTH=1 to run without a login (local testing only)."
            )

    def _fingerprint(self) -> str:
        # Ties a session to the current credentials: changing them invalidates it.
        msg = f"{self.user}\0{self.password}".encode("utf-8")
        return hmac.new(self._secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()[:32]

    def check_credentials(self, user: str, password: str) -> bool:
        if not self.enabled:
            return False
        user_ok = hmac.compare_digest(user.encode("utf-8"), self.user.encode("utf-8"))
        password_ok = hmac.compare_digest(password.encode("utf-8"), self.password.encode("utf-8"))
        return user_ok & password_ok

    def issue_token(self) -> str:
        return self._serializer.dumps({"sid": secrets.token_hex(8), "fp": self._fingerprint()})

    def _load_token(self, token: str):
        if not token or not self.enabled:
            return None
        try:
            data = self._serializer.loads(token, max_age=self.max_age_s)
        except BadSignature:  # also covers SignatureExpired
            return None
        if not isinstance(data, dict) or not hmac.compare_digest(str(data.get("fp", "")), self._fingerprint()):
            return None
        return data

    def is_valid(self, token: str) -> bool:
        data = self._load_token(token)
        return data is not None and data.get("sid") not in self._revoked

    def revoke(self, token: str) -> None:
        data = self._load_token(token)
        if data:
            self._revoked.add(data.get("sid"))

    def _recent_failures(self, client: str):
        cutoff = time.monotonic() - FAILURE_WINDOW_S
        recent = [t for t in self._failures.get(client, []) if t > cutoff]
        if recent:
            self._failures[client] = recent
        else:
            self._failures.pop(client, None)
        return recent

    def is_rate_limited(self, client: str) -> bool:
        return len(self._recent_failures(client)) >= MAX_FAILURES

    def record_failure(self, client: str) -> None:
        if len(self._failures) > 10000:
            self._failures.clear()
        self._failures.setdefault(client, []).append(time.monotonic())

    def clear_failures(self, client: str) -> None:
        self._failures.pop(client, None)


def build_auth_router(auth: AuthManager, login_page_path: str) -> APIRouter:
    router = APIRouter()

    def login_page(message: str = "", status_code: int = 200) -> HTMLResponse:
        with open(login_page_path, "r", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(html.replace("<!--MESSAGE-->", escape(message)), status_code=status_code,
                            headers={"Cache-Control": "no-store"})

    @router.get("/login", response_class=HTMLResponse)
    async def get_login(request: Request):
        if auth.open_access or auth.is_valid(request.cookies.get(COOKIE_NAME, "")):
            return RedirectResponse("/", status_code=303)
        return login_page()

    @router.post("/login")
    async def post_login(request: Request):
        if auth.open_access:
            return RedirectResponse("/", status_code=303)
        client = request.client.host if request.client else "unknown"
        form = await request.form()
        username, password = form.get("username", ""), form.get("password", "")
        # No await between the limit check and recording the result.
        if auth.is_rate_limited(client):
            return login_page("Too many failed attempts. Try again in a few minutes.", 429)
        if not (isinstance(username, str) and isinstance(password, str) and auth.check_credentials(username, password)):
            auth.record_failure(client)
            return login_page("Wrong username or password.", 401)
        auth.clear_failures(client)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(COOKIE_NAME, auth.issue_token(), max_age=auth.max_age_s, path="/",
                            httponly=True, samesite="lax", secure=auth.cookie_secure)
        return response

    @router.api_route("/logout", methods=["GET", "POST"])
    async def logout(request: Request):
        auth.revoke(request.cookies.get(COOKIE_NAME, ""))
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE_NAME, path="/", httponly=True, samesite="lax", secure=auth.cookie_secure)
        return response

    return router


class SecurityMiddleware:
    """Authentication, same-origin check, request size limits and security headers."""

    def __init__(self, app, auth: AuthManager, body_limits: dict, default_body_limit: int):
        self.app = app
        self.auth = auth
        self.body_limits = body_limits
        self.default_body_limit = default_body_limit

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            return await self.app(scope, receive, send)

        conn = HTTPConnection(scope)
        path = scope["path"]
        allowed = (
            path in PUBLIC_PATHS
            or self.auth.open_access
            or self.auth.is_valid(conn.cookies.get(COOKIE_NAME, ""))
        )

        if scope["type"] != "http":  # websockets: the app defines none
            if not allowed:
                return await send({"type": "websocket.close", "code": 1008})
            return await self.app(scope, receive, send)

        method = scope["method"]
        headers = Headers(scope=scope)

        if not allowed:
            if method in ("GET", "HEAD") and not path.startswith("/api/"):
                response = RedirectResponse("/login", status_code=303)
            else:
                response = JSONResponse({"detail": "Not authenticated."}, status_code=401)
            return await response(scope, receive, send)

        # Browsers send Origin on cross-site writes; reject mismatches.
        origin = headers.get("origin")
        if method not in SAFE_METHODS and origin and urlsplit(origin).netloc != headers.get("host", ""):
            response = JSONResponse({"detail": "Cross-origin request rejected."}, status_code=403)
            return await response(scope, receive, send)

        limit = self.body_limits.get(path, self.default_body_limit)
        too_large = JSONResponse(
            {"detail": f"Request too large (limit {limit / (1024 * 1024):.1f} MB)."}, status_code=413)
        length = headers.get("content-length", "")
        if length.isdigit() and int(length) > limit:
            return await too_large(scope, receive, send)

        state = {"received": 0, "overflow": False, "started": False, "replaced": False}

        async def limited_receive():
            message = await receive()
            if message["type"] == "http.request":
                state["received"] += len(message.get("body", b""))
                if state["received"] > limit:
                    state["overflow"] = True
                    # End the body here so nothing more is buffered or spooled.
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def secured_send(message):
            if state["replaced"]:
                return
            if message["type"] == "http.response.start":
                if state["overflow"] and not state["started"]:
                    state["replaced"] = True
                    return await too_large(scope, receive, send)
                state["started"] = True
                out = MutableHeaders(scope=message)
                out["X-Content-Type-Options"] = "nosniff"
                out["X-Frame-Options"] = "DENY"
                out["Referrer-Policy"] = "same-origin"
                if not path.startswith("/static/") and "cache-control" not in out:
                    out["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, limited_receive, secured_send)
