#!/usr/bin/env python3
"""
Rewrite policy file paths embedded in `policy_eval_cache.pkl`.

This repo's policy evaluation cache uses a string key like:
  env=...|dim=...|...|method=...|path=/abs/path/to/policy.pth(|mtime=...)

If you moved policy files to a new directory, you can rewrite the `path=...`
portion in all keys, keeping cached evaluation results.

Design goals:
- Only asks the user for OLD prefix and NEW prefix (interactive).
- Creates a timestamped backup next to the cache file.
- Detects key collisions after rewrite.
- Optionally updates `mtime=` in the key to match the new path's mtime (default: yes).
"""

from __future__ import annotations

import datetime as _dt
import os
import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[1] / "cache" / "policy_eval_cache.pkl"


@dataclass(frozen=True)
class RewriteStats:
    total_entries: int
    touched_entries: int
    rewritten_entries: int
    unchanged_entries: int
    missing_new_files: int
    collisions: int


def _normalize_prefix(p: str) -> str:
    """
    Normalize a user-entered prefix to improve matching.
    - Expands ~
    - Converts to absolute path
    - Removes trailing slash (except root)
    """
    expanded = os.path.expanduser(p.strip())
    abs_p = os.path.abspath(expanded)
    if abs_p != os.sep:
        abs_p = abs_p.rstrip(os.sep)
    return abs_p


def _rewrite_key(
    key: str,
    old_prefix: str,
    new_prefix: str,
    update_mtime: bool,
) -> Tuple[str, bool, bool]:
    """
    Returns: (new_key, changed, missing_new_file)
    """
    if "path=" not in key:
        return key, False, False

    parts = key.split("|")
    changed = False
    missing_new_file = False
    new_abs_path: str | None = None

    for i, part in enumerate(parts):
        if part.startswith("path="):
            old_path = part[len("path=") :]
            # The cache uses absolute path. We still guard just in case.
            old_path_abs = os.path.abspath(os.path.expanduser(old_path))
            if old_path_abs.startswith(old_prefix):
                rewritten = new_prefix + old_path_abs[len(old_prefix) :]
                parts[i] = "path=" + rewritten
                new_abs_path = rewritten
                changed = True
            else:
                # not under old prefix -> untouched
                new_abs_path = old_path_abs

    if update_mtime and changed and new_abs_path is not None:
        # Only update when key actually changed.
        try:
            mtime = os.path.getmtime(new_abs_path)
        except Exception:
            mtime = None
            missing_new_file = True

        if mtime is not None:
            for i, part in enumerate(parts):
                if part.startswith("mtime="):
                    parts[i] = "mtime=" + str(mtime)
                    break
            # If no mtime= field exists, do nothing (HC/Ant style keys).

    new_key = "|".join(parts)
    return new_key, changed, missing_new_file


def rewrite_cache_keys_inplace(
    cache_path: Path,
    old_prefix: str,
    new_prefix: str,
    update_mtime: bool = True,
) -> RewriteStats:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    # Load
    with cache_path.open("rb") as f:
        cache = pickle.load(f)

    if not isinstance(cache, dict):
        raise TypeError(f"Expected a dict in cache, got: {type(cache)}")

    total = len(cache)
    touched = 0  # keys with 'path='
    rewritten = 0
    unchanged = 0
    missing_new_files = 0
    collisions = 0

    new_cache: Dict[Any, Any] = {}
    seen_new_keys = set()

    for k, v in cache.items():
        if isinstance(k, str) and "path=" in k:
            touched += 1
            new_k, changed, missing = _rewrite_key(
                k, old_prefix=old_prefix, new_prefix=new_prefix, update_mtime=update_mtime
            )
            if missing:
                missing_new_files += 1
            if changed:
                rewritten += 1
            else:
                unchanged += 1
        else:
            new_k = k

        if new_k in seen_new_keys:
            collisions += 1
            # Keep the first occurrence and drop the later one to avoid silently overwriting.
            # (If you want different behavior, edit here.)
            continue

        seen_new_keys.add(new_k)
        new_cache[new_k] = v

    # Backup
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = cache_path.with_suffix(cache_path.suffix + f".bak_{ts}")
    shutil.copy2(cache_path, backup_path)

    # Save
    with cache_path.open("wb") as f:
        pickle.dump(new_cache, f)

    return RewriteStats(
        total_entries=total,
        touched_entries=touched,
        rewritten_entries=rewritten,
        unchanged_entries=unchanged,
        missing_new_files=missing_new_files,
        collisions=collisions,
    )


def _prompt_two_prefixes() -> Tuple[str, str]:
    print("policy_eval_cache.pkl 内の `path=...` を一括置換します。")
    old_prefix = input("旧パスprefix（例: /path/to/workspace/exp ）: ").strip()
    new_prefix = input("新パスprefix（例: /path/to/workspace/exp ）: ").strip()
    if not old_prefix or not new_prefix:
        raise ValueError("旧パスprefix / 新パスprefix は空にできません。")
    return old_prefix, new_prefix


def main(argv: Iterable[str]) -> int:
    # Minimal CLI: optionally allow overriding cache path, but keep interactive inputs.
    args = list(argv)
    cache_path = DEFAULT_CACHE_PATH
    update_mtime = True

    # Accept:
    #   --cache /path/to/policy_eval_cache.pkl
    #   --no-mtime
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--cache":
            i += 1
            if i >= len(args):
                raise ValueError("--cache requires a path argument")
            cache_path = Path(args[i])
        elif a == "--no-mtime":
            update_mtime = False
        else:
            raise ValueError(f"Unknown arg: {a}")
        i += 1

    old_prefix_raw, new_prefix_raw = _prompt_two_prefixes()
    old_prefix = _normalize_prefix(old_prefix_raw)
    new_prefix = _normalize_prefix(new_prefix_raw)

    print("\n--- 確認 ---")
    print(f"cache: {cache_path}")
    print(f"old_prefix: {old_prefix}")
    print(f"new_prefix: {new_prefix}")
    print(f"update_mtime: {update_mtime}")
    yn = input("この内容で書き換えますか？ [y/N]: ").strip().lower()
    if yn != "y":
        print("中止しました。")
        return 1

    stats = rewrite_cache_keys_inplace(
        cache_path=cache_path,
        old_prefix=old_prefix,
        new_prefix=new_prefix,
        update_mtime=update_mtime,
    )

    print("\n--- 完了 ---")
    print(f"total entries: {stats.total_entries}")
    print(f"touched (has path=): {stats.touched_entries}")
    print(f"rewritten: {stats.rewritten_entries}")
    print(f"unchanged (path= but not under old_prefix): {stats.unchanged_entries}")
    print(f"missing new files (mtime update failed): {stats.missing_new_files}")
    print(f"collisions (dropped later entries): {stats.collisions}")
    print("\n※ 同じフォルダに .bak_YYYYMMDD_HHMMSS を作ってあります（復元用）。")

    if stats.collisions > 0:
        print(
            "⚠ キー衝突が発生しています。旧prefixが広すぎる/新prefixが間違っている可能性があります。\n"
            "  まずはバックアップから復元して、prefixを見直してください。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

