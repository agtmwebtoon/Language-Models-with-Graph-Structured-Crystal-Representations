from __future__ import annotations
from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
from datetime import datetime

def _to_serializable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj

def dump_config_txt(cfg, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Config snapshot ({datetime.now().isoformat(timespec='seconds')})")
    lines.append("")
    lines.append("[paths]")
    for k, v in asdict(cfg.paths).items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("[model]")
    for k, v in asdict(cfg.model).items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("[training]")
    for k, v in asdict(cfg.training).items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("[tokenizer]")
    for k, v in asdict(cfg.tokenizer).items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("[visualization]")
    for k, v in asdict(cfg.visualization).items():
        lines.append(f"{k}: {v}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def dump_config_json(cfg, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "paths": asdict(cfg.paths),
        "model": asdict(cfg.model),
        "training": asdict(cfg.training),
        "tokenizer": asdict(cfg.tokenizer),
        "visualization": asdict(cfg.visualization),
    }
    # Path가 있으면 문자열로
    payload = json.loads(json.dumps(payload, default=str))
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
