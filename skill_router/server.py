"""本地演示服务：把三层链路的中间过程摊开给网页看。

    python run_server.py --skills data/clawhub --port 8000

只用标准库 —— 整条链路的核心是零依赖的，演示服务不该反过来引入框架。

    GET  /             单页前端
    GET  /api/meta     技能库与链路的配置信息
    POST /api/route    跑一次完整路由，返回各层中间数据

``POST /api/route`` 的请求体是 ``{"query": "...", "top_k": 3}``，``top_k`` 可选。

**拒答阈值不接受覆盖** —— 它是标定出来的链路参数（``MIN_SCORE``），放给客户端调
等于让调用方绕过标定结果。推荐条数不同：它只决定「最多给几条」，属于展示偏好，
允许逐次指定。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Sequence
from urllib.parse import urlparse

from .config import load_env
from .router import Router, build_router

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(PROJECT_ROOT, "web")
INDEX_FILE = os.path.join(WEB_DIR, "index.html")


class Handler(BaseHTTPRequestHandler):
    server_version = "SkillRouter/0.1"

    # 由 serve() 注入：Router 无状态，全局共用一个实例即可
    router: Optional[Router] = None
    skills_dir: str = ""

    # -- 响应工具 ---------------------------------------------------------

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _read_json(self) -> Optional[Dict[str, Any]]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    # -- 路由分发 ---------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - 标准库约定的方法名
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_index()
        elif path == "/api/meta":
            self._send_json(200, self._meta())
        else:
            self._send_json(404, {"error": f"未知路径 {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/route":
            self._send_json(404, {"error": f"未知路径 {path}"})
            return

        payload = self._read_json()
        if payload is None:
            self._send_json(400, {"error": "请求体不是合法 JSON 对象"})
            return

        query = str(payload.get("query") or "").strip()
        if not query:
            self._send_json(400, {"error": "query 不能为空"})
            return

        if self.router is None:
            self._send_json(503, {"error": "技能库未就绪"})
            return

        # min_score 归服务端配置，客户端无权覆盖；top_k 只影响展示条数，允许指定
        options: Dict[str, Any] = {}
        if payload.get("top_k") is not None:
            try:
                options["top_k"] = max(1, int(payload["top_k"]))
            except (TypeError, ValueError):
                self._send_json(400, {"error": "top_k 必须是整数"})
                return

        try:
            result = self.router.route(query, **options)
        except Exception as error:  # noqa: BLE001 - 兜底：一个请求不该打挂整个服务
            self._send_json(500, {"error": f"链路异常：{error}"})
            return

        self._send_json(200, result.to_dict())

    # -- 具体响应 ---------------------------------------------------------

    def _serve_index(self) -> None:
        # 每次读盘：本地开发时改前端不用重启服务
        try:
            with open(INDEX_FILE, "rb") as handle:
                body = handle.read()
        except OSError:
            self._send(500, "text/plain; charset=utf-8", "缺少 web/index.html".encode("utf-8"))
            return
        self._send(200, "text/html; charset=utf-8", body)

    def _meta(self) -> Dict[str, Any]:
        router = self.router
        if router is None:
            return {"ready": False}
        return {
            "ready": True,
            "skills_dir": self.skills_dir,
            "library_size": len(router.skills),
            "sparse_k": router.recaller.sparse_k,
            "dense_k": router.recaller.dense_k,
            "small_n": router.recaller.small_n,
            "top_k": router.top_k,
            "min_score": router.min_score,
            "judge": type(router.judge).__name__,
            "judge_model": str(getattr(router.judge, "model", "-")),
            "dense_model": router.dense_model,
        }

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")


def serve(
    skills_dir: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    top_k: int = 3,
    min_score: float = 0.0,
    small_n: int = 60,
    dense: Optional[bool] = None,
) -> None:
    router = build_router(
        skills_dir, top_k=top_k, min_score=min_score, small_n=small_n, dense=dense
    )
    if not router.skills:
        print(f"技能库为空：{skills_dir}", file=sys.stderr)
        raise SystemExit(2)

    print(f"技能库  : {skills_dir}（{len(router.skills)} 个技能）")
    print(
        f"召回    : 稀疏 {router.recaller.sparse_k} + 稠密 {router.recaller.dense_k}"
        f"（{router.dense_model or '未接入，纯 BM25'}），库规模阈值 {router.recaller.small_n}"
    )
    print(f"精排    : {type(router.judge).__name__} / {getattr(router.judge, 'model', '-')}")
    print("预热稠密向量 …", end="", flush=True)
    started = time.perf_counter()
    router.warmup()
    print(f" 完成（{int((time.perf_counter() - started) * 1000)} ms）")

    Handler.router = router
    Handler.skills_dir = skills_dir

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"\n服务已启动：http://{host}:{port}   按 Ctrl+C 停止。")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_env()

    parser = argparse.ArgumentParser(
        prog="python run_server.py",
        description="启动本地演示服务",
    )
    parser.add_argument("--skills", default=os.environ.get("SKILLS_DIR", "data/skills"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", 3)))
    parser.add_argument(
        "--min-score", type=float, default=float(os.environ.get("MIN_SCORE", "0.4"))
    )
    parser.add_argument("--small-n", type=int, default=int(os.environ.get("SMALL_N", 60)))
    parser.add_argument(
        "--dense", action="store_true", default=os.environ.get("USE_EMBED") == "1"
    )
    parser.add_argument("--no-dense", dest="dense", action="store_false")
    args = parser.parse_args(argv)

    serve(
        args.skills,
        host=args.host,
        port=args.port,
        top_k=args.top_k,
        min_score=args.min_score,
        small_n=args.small_n,
        dense=args.dense,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
