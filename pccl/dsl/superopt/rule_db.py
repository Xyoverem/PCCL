"""Rule cache management under ~/.pccl/superopt/."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from .rule import RewriteRule


def _cache_dir() -> Path:
    return Path.home() / ".pccl" / "superopt"


def _profiles_dir() -> Path:
    return _cache_dir() / "profiles"


def _ensure_dirs():
    _cache_dir().mkdir(parents=True, exist_ok=True)
    _profiles_dir().mkdir(parents=True, exist_ok=True)


def save_rules(k: int, rules: List[RewriteRule]) -> Path:
    _ensure_dirs()
    path = _cache_dir() / f"rules_k{k}.json"
    data = [r.to_dict() for r in rules]
    path.write_text(json.dumps(data, indent=2))
    return path


def load_rules(k: int) -> List[RewriteRule]:
    path = _cache_dir() / f"rules_k{k}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [RewriteRule.from_dict(d) for d in data]


def load_all_rules(max_k: int = 6) -> List[RewriteRule]:
    all_rules = []
    for k in range(1, max_k + 1):
        all_rules.extend(load_rules(k))
    return all_rules


def save_profile(name: str, profile: Dict[str, Any]) -> Path:
    _ensure_dirs()
    path = _profiles_dir() / f"{name}.json"
    path.write_text(json.dumps(profile, indent=2))
    return path


def load_profile(name: str) -> Optional[Dict[str, Any]]:
    path = _profiles_dir() / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def clear_rules(k: Optional[int] = None):
    if k is not None:
        path = _cache_dir() / f"rules_k{k}.json"
        if path.exists():
            path.unlink()
    else:
        for f in _cache_dir().glob("rules_k*.json"):
            f.unlink()
