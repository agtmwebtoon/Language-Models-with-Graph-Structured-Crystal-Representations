# src/utils/run_naming.py
from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, Optional
import re
from pathlib import Path

def _slug(s: str) -> str:
    s = s.strip().lower().replace("/", "-")
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "run"

def build_run_name(cfg) -> str:
    m = cfg.model
    text_id = _slug(str(m.text_model_id))
    graph_id = _slug(str(m.graph_backbone_id))
    return f"{text_id}_{graph_id}"

def resolve_run_dir_with_suffix(runs_dir: Path, base_name: str) -> Path:
    """
    If runs_dir/base_name exists, create runs_dir/base_name_001, _002, ...
    Returns a directory path that does not yet exist.
    """
    runs_dir = Path(runs_dir)
    base = runs_dir / base_name

    if not base.exists():
        return base

    i = 1
    while True:
        cand = runs_dir / f"{base_name}_{i:03d}"
        if not cand.exists():
            return cand
        i += 1

def select_latest_run_dir(runs_dir: Path, base_name: str) -> Optional[Path]:
    """
    Select latest run directory among:
      base_name
      base_name_001
      base_name_002
      ...

    Rule:
      - base_name        -> index 0
      - base_name_XXX    -> index XXX
      - choose max index
    """
    runs_dir = Path(runs_dir)
    pattern = re.compile(rf"^{re.escape(base_name)}(?:_(\d+))?$")

    candidates = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            idx = int(m.group(1)) if m.group(1) is not None else 0
            candidates.append((idx, d))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]