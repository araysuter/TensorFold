"""Validated, local-only configuration for the Studio model gateway.

Configuration contains paths to credentials, never credential values. The JSON
file belongs outside the source checkout, normally in ~/.config/tensorfold/.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path


MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}\Z")
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


@dataclasses.dataclass(frozen=True)
class KeyFile:
    label: str
    path: Path
    required: bool = True


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    id: str
    argv: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    readiness_timeout: float = 240.0


@dataclasses.dataclass(frozen=True)
class GatewayConfig:
    models: tuple[ModelConfig, ...]
    key_files: tuple[KeyFile, ...]
    cache_root: Path
    host: str = "127.0.0.1"
    port: int = 8080
    backend_port: int = 18080
    default_model: str = "swift-1.5"
    idle_seconds: float = 1800.0
    max_parked: int = 4
    max_body_bytes: int = 8 * 1024 * 1024
    max_pending: int = 16
    queue_timeout: float = 3600.0
    request_timeout: float = 3600.0
    stop_timeout: float = 20.0
    release_delay: float = 1.0
    allowed_hosts: tuple[str, ...] = ()


def _path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("configuration path must be nonempty text")
    return Path(value).expanduser().absolute()


def validate(config: GatewayConfig) -> GatewayConfig:
    if config.host != "127.0.0.1":
        raise ValueError("gateway must bind 127.0.0.1; use the existing authenticated tunnel")
    for port in (config.port, config.backend_port):
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("ports must be integers from 0 through 65535")
    if config.backend_port == 0 or config.port == config.backend_port:
        raise ValueError("backend must use a separate, fixed loopback port")
    if not config.models or not config.key_files:
        raise ValueError("at least one model and one API key file are required")
    names: set[str] = set()
    for model in config.models:
        for name in (model.id, *model.aliases):
            if not isinstance(name, str) or not MODEL_ID.fullmatch(name) or name in names:
                raise ValueError("model IDs and aliases must be valid and unique")
            names.add(name)
        if not model.argv or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in model.argv):
            raise ValueError("each model needs an executable argv array; shell commands are unsupported")
        if not Path(model.argv[0]).expanduser().is_absolute():
            raise ValueError("model executable must be an absolute path")
        # These substitutions are exclusively supplied by the gateway. Clients
        # cannot select an executable, environment variable, or snapshot path.
        for arg in model.argv:
            for placeholder in re.findall(r"\{([^{}]+)\}", arg):
                if placeholder not in {"port", "backend_port", "backend_key_file", "snapshot_dir"}:
                    raise ValueError("unknown model argv placeholder")
        if not isinstance(model.env, dict) or any(
            not isinstance(k, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)
            or not isinstance(v, str) or "\0" in v for k, v in model.env.items()
        ):
            raise ValueError("model environment must contain valid string entries")
        if not 0 < model.readiness_timeout <= 1800:
            raise ValueError("model readiness timeout must be between zero and 1800 seconds")
    if config.default_model not in names:
        raise ValueError("default model must name a configured model")
    for key in config.key_files:
        if not isinstance(key.label, str) or not LABEL.fullmatch(key.label):
            raise ValueError("key file labels must use letters, numbers, dots, underscores or hyphens")
        if not isinstance(key.path, Path) or not key.path.is_absolute():
            raise ValueError("key file paths must be absolute")
        if not isinstance(key.required, bool):
            raise ValueError("key file required must be a boolean")
    if not isinstance(config.cache_root, Path) or not config.cache_root.is_absolute():
        raise ValueError("cache_root must be an absolute dedicated directory")
    # Restrict automated deletion to a dedicated directory, never HOME, /, a
    # model directory, or an arbitrary broad parent. Runtime deletes its own
    # run-* children only, and does not delete cache_root itself.
    if config.cache_root == Path.home() or len(config.cache_root.parts) < 3:
        raise ValueError("cache_root must be a dedicated directory below a parent")
    for key, low, high in (
        ("idle_seconds", 0.05, 86400), ("queue_timeout", 0.05, 86400),
        ("request_timeout", 0.05, 86400), ("stop_timeout", 0.05, 120),
        ("release_delay", 0, 30),
    ):
        value = getattr(config, key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f"invalid {key}")
    for key, low, high in (
        ("max_parked", 0, 4), ("max_body_bytes", 1024, 32 * 1024 * 1024),
        ("max_pending", 1, 128),
    ):
        value = getattr(config, key)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"invalid {key}")
    if any(not isinstance(host, str) or not host or any(ch.isspace() for ch in host)
           or "/" in host or "\\" in host for host in config.allowed_hosts):
        raise ValueError("allowed_hosts must be hostnames without whitespace or paths")
    return config


def load_config(path: str | Path) -> GatewayConfig:
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("gateway configuration must be a JSON object")
    allowed = {field.name for field in dataclasses.fields(GatewayConfig)}
    if document.keys() - allowed:
        raise ValueError("unknown gateway configuration field")
    if (not isinstance(document.get("models"), list)
            or not all(isinstance(row, dict) and isinstance(row.get("argv"), list)
                       and isinstance(row.get("aliases", []), list)
                       and not (row.keys() - {field.name for field in dataclasses.fields(ModelConfig)})
                       for row in document["models"])):
        raise ValueError("models must be objects with argv arrays")
    if (not isinstance(document.get("key_files"), list)
            or not all(isinstance(row, dict)
                       and not (row.keys() - {field.name for field in dataclasses.fields(KeyFile)})
                       for row in document["key_files"])):
        raise ValueError("key_files must be objects with paths")
    document["cache_root"] = _path(document["cache_root"])
    document["models"] = tuple(ModelConfig(
        id=row["id"], argv=tuple(os.path.expanduser(arg) for arg in row["argv"]),
        aliases=tuple(row.get("aliases", ())), env=dict(row.get("env", {})),
        readiness_timeout=row.get("readiness_timeout", 240),
    ) for row in document["models"])
    document["key_files"] = tuple(KeyFile(
        label=row["label"], path=_path(row["path"]), required=row.get("required", True),
    ) for row in document["key_files"])
    if "allowed_hosts" in document:
        document["allowed_hosts"] = tuple(document["allowed_hosts"])
    return validate(GatewayConfig(**document))
