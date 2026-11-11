"""YAML configs: every run is configs/base.yaml with one small file merged on top."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``section.key=value`` strings (values are parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        dotted, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override {item!r} is not key=value")
        node = cfg
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    if path is not None:
        cfg = deep_merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    cfg = apply_overrides(cfg, overrides or [])
    if "name" not in cfg:
        raise ValueError("config needs a 'name' (used as the results folder)")
    return cfg
