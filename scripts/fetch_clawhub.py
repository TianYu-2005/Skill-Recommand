#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 ClawHub 批量下载技能。

ClawHub 的公共只读接口**不需要 token**：

    GET /api/v1/skills?limit=&cursor=&sort=   技能列表（游标翻页）
    GET /api/v1/download?slug=&version=       技能打包（ZIP 字节）

本脚本只做三件事：拉列表、下 ZIP、解压落盘，并把来源与版本记进 `_meta.json`。

**不执行技能中的任何脚本** —— 第三方技能是不可信输入，落地后仍需人工审查
（`SKILL.md` 与 `scripts/`）。ClawHub 另有 `/api/v1/skills/{slug}/scan` 端点
可直接取安全扫描结果，需要时自行调用。

用法：

    python scripts/fetch_clawhub.py --count 200
    python scripts/fetch_clawhub.py --count 20 --sort updated
    python scripts/fetch_clawhub.py --count 200 --out data/clawhub
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

USER_AGENT = "skill-recommand/0.1 (+https://github.com/TianYu-2005/Skill-Recommand)"
SORTS = ("downloads", "updated", "recommended", "createdAt", "stars")
PAGE_LIMIT = 50


def _get(url: str, timeout: int = 60, retries: int = 4) -> bytes:
    """GET 并返回原始字节。429 / 5xx 按 Retry-After 与指数退避重试。"""
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code in (429, 500, 502, 503, 504):
                wait = float(error.headers.get("Retry-After") or 2 ** attempt)
                time.sleep(min(wait, 30))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"请求失败: {url} ({last_error})")


def list_skills(registry: str, count: int, sort: str) -> List[Dict[str, Any]]:
    """翻页取前 ``count`` 个技能。"""
    items: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while len(items) < count:
        query: Dict[str, Any] = {"limit": min(PAGE_LIMIT, count - len(items)), "sort": sort}
        if cursor:
            query["cursor"] = cursor
        payload = json.loads(_get(f"{registry}/api/v1/skills?{urllib.parse.urlencode(query)}"))

        batch = payload.get("items") or []
        items.extend(batch)

        cursor = payload.get("nextCursor")
        if not batch or not cursor:
            break

    return items[:count]


def _safe_name(slug: str) -> str:
    """把 slug 变成安全的目录名（防止路径穿越）。"""
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in slug)
    return cleaned.strip("._") or "unknown"


def _extract(payload: bytes, target: str) -> int:
    """解压 ZIP 到目标目录，返回文件数。拒绝任何越界路径。"""
    root = os.path.realpath(target)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            resolved = os.path.realpath(os.path.join(root, member.filename))
            if resolved != root and not resolved.startswith(root + os.sep):
                raise ValueError(f"压缩包含越界路径: {member.filename}")
        archive.extractall(target)
        return len(archive.infolist())


def _fetch_one(item: Dict[str, Any], registry: str, out_dir: str) -> Dict[str, Any]:
    """下载并解压单个技能。已落地则跳过，便于断点续传。"""
    slug = item.get("slug") or ""
    owner = item.get("ownerHandle") or ""
    record: Dict[str, Any] = {
        "slug": slug,
        "owner": owner,
        "displayName": item.get("displayName"),
        "summary": item.get("summary"),
        "downloads": (item.get("stats") or {}).get("downloads"),
    }

    if not slug:
        record["status"] = "invalid"
        return record

    version = (item.get("latestVersion") or {})
    record["version"] = version.get("version")
    record["license"] = version.get("license")

    # slug 会重名（热门技能尤其如此，如多个发布者都有 "github"），
    # 必须带上 ownerHandle 才唯一；目录名同样按 owner__slug 消歧。
    folder = _safe_name(f"{owner}__{slug}" if owner else slug)
    record["folder"] = folder
    target = os.path.join(out_dir, folder)
    if os.path.isfile(os.path.join(target, "SKILL.md")):
        record["status"] = "skipped"
        return record

    query: Dict[str, Any] = {"slug": slug}
    if owner:
        query["ownerHandle"] = owner
    try:
        payload = _get(f"{registry}/api/v1/download?{urllib.parse.urlencode(query)}", timeout=120)
    except Exception as error:  # noqa: BLE001 - 单个失败不该中断整批
        record["status"] = "failed"
        record["error"] = f"下载失败: {error}"
        return record

    # 少数由 GitHub 托管的技能不返回 ZIP，而是移交描述符
    if payload[:2] != b"PK":
        record["status"] = "handoff"
        record["error"] = "非 ZIP 响应（GitHub 移交描述符）"
        with open(os.path.join(out_dir, f"{folder}.handoff.json"), "wb") as handle:
            handle.write(payload)
        return record

    try:
        os.makedirs(target, exist_ok=True)
        record["files"] = _extract(payload, target)
    except Exception as error:  # noqa: BLE001
        record["status"] = "failed"
        record["error"] = f"解压失败: {error}"
        return record

    if not os.path.isfile(os.path.join(target, "SKILL.md")):
        record["status"] = "no-skill-md"
        return record

    record["status"] = "ok"
    record["bytes"] = len(payload)
    return record


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/fetch_clawhub.py",
        description="从 ClawHub 批量下载技能（公共只读接口，无需 token）",
    )
    parser.add_argument("--count", type=int, default=200, help="下载多少个技能，默认 200")
    parser.add_argument("--sort", default="downloads", choices=SORTS, help="列表排序，默认 downloads")
    parser.add_argument("--out", default=os.path.join("data", "clawhub"), help="输出目录")
    parser.add_argument("--workers", type=int, default=8, help="并发下载数，默认 8")
    parser.add_argument(
        "--registry",
        default=os.environ.get("CLAWHUB_REGISTRY", "https://clawhub.ai"),
        help="注册中心地址",
    )
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    print(f"注册中心 : {args.registry}")
    print(f"排序     : {args.sort}")
    print(f"目标数量 : {args.count}")
    print(f"输出目录 : {args.out}")
    print("-" * 72)

    try:
        items = list_skills(args.registry, args.count, args.sort)
    except Exception as error:  # noqa: BLE001
        print(f"拉取技能列表失败：{error}", file=sys.stderr)
        return 2

    if not items:
        print("列表为空。", file=sys.stderr)
        return 2
    print(f"列表已取到 {len(items)} 个技能，开始下载…")

    records: List[Dict[str, Any]] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch_one, item, args.registry, args.out): item for item in items}
        for done, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            if record["status"] != "ok":
                print(f"[{done}/{len(items)}] {record['status']:<12} {record['slug']}", flush=True)
            elif done % 20 == 0 or done == len(items):
                print(f"[{done}/{len(items)}] 已处理…", flush=True)

    summary: Dict[str, int] = {}
    for record in records:
        summary[record["status"]] = summary.get(record["status"], 0) + 1

    meta = {
        "registry": args.registry,
        "sort": args.sort,
        "fetchedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requested": args.count,
        "summary": summary,
        "skills": sorted(records, key=lambda row: row["slug"]),
    }
    meta_path = os.path.join(args.out, "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("-" * 72)
    print(f"耗时     : {time.time() - started:.1f}s")
    for status, number in sorted(summary.items()):
        print(f"{status:<12} {number}")
    print(f"来源记录 : {meta_path}")

    return 0 if summary.get("ok", 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
