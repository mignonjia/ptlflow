"""Utilities for resolving checkpoint paths."""

from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from torch import hub


def is_hf_checkpoint_path(ckpt_path: str) -> bool:
    """Return True when ckpt_path points to the Hugging Face Hub."""
    return ckpt_path.startswith(("hf://", "hf:")) or ckpt_path.startswith(
        ("https://huggingface.co/", "http://huggingface.co/")
    )


def is_url(ckpt_path: str) -> bool:
    """Return True when ckpt_path is an HTTP(S) URL."""
    parsed = urlparse(ckpt_path)
    return parsed.scheme in {"http", "https"}


def resolve_checkpoint_path(
    ckpt_path: str,
    model_ref=None,
    force_local: bool = False,
) -> str:
    """Resolve a checkpoint path or alias.

    Parameters
    ----------
    ckpt_path : str
        Checkpoint path or alias.
    model_ref : optional
        Reference to the model class.
    force_local : bool, default False
        If True, remote URLs are downloaded before loading.
    """
    resolved_path = ckpt_path
    if Path(resolved_path).exists():
        return str(Path(resolved_path))

    if model_ref is not None and hasattr(model_ref, "pretrained_checkpoints"):
        tmp_ckpt_path = model_ref.pretrained_checkpoints.get(resolved_path)
        if tmp_ckpt_path is not None:
            resolved_path = tmp_ckpt_path
        elif not is_hf_checkpoint_path(resolved_path) and not is_url(resolved_path):
            raise ValueError(
                f"Invalid checkpoint name {resolved_path}. "
                f'Choose one from {{{",".join(model_ref.pretrained_checkpoints.keys())}}}'
            )
    elif not is_hf_checkpoint_path(resolved_path) and not is_url(resolved_path):
        model_name = model_ref.__name__ if model_ref is not None else "unknown"
        raise ValueError(
            f"Cannot find checkpoint {resolved_path} for model {model_name}"
        )

    if is_hf_checkpoint_path(resolved_path):
        return download_hf_checkpoint(resolved_path)
    if force_local and is_url(resolved_path):
        return download_remote_checkpoint(resolved_path)
    return resolved_path


def download_hf_checkpoint(ckpt_path: str) -> str:
    """Download a checkpoint from the Hugging Face Hub."""
    repo_id, filename, revision = _parse_hf_checkpoint_path(ckpt_path)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Hugging Face checkpoint support requires `huggingface_hub`. "
            "Install PTLFlow with `pip install ptlflow[hf]` or install "
            "`huggingface_hub` directly."
        ) from exc

    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def download_remote_checkpoint(ckpt_url: str) -> str:
    """Download a checkpoint URL into the torch hub cache."""
    parsed = urlparse(ckpt_url)
    filename = Path(unquote(parsed.path)).name
    model_dir = Path(hub.get_dir()) / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    dst_path = model_dir / filename
    if not dst_path.exists():
        hub.download_url_to_file(ckpt_url, str(dst_path))
    return str(dst_path)


def _parse_hf_checkpoint_path(ckpt_path: str) -> Tuple[str, str, Optional[str]]:
    if ckpt_path.startswith(("https://huggingface.co/", "http://huggingface.co/")):
        return _parse_hf_url(ckpt_path)
    return _parse_hf_short_path(ckpt_path)


def _parse_hf_short_path(ckpt_path: str) -> Tuple[str, str, Optional[str]]:
    stripped_path = ckpt_path[5:] if ckpt_path.startswith("hf://") else ckpt_path[3:]
    parsed = urlparse(f"//{stripped_path}")
    path_parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(path_parts) < 2:
        raise ValueError(
            "Invalid Hugging Face checkpoint path. Use "
            "`hf://<namespace>/<repo>/<path/to/file.ckpt>`."
        )

    repo_id = f"{parsed.netloc}/{path_parts[0]}" if parsed.netloc else path_parts[0]
    filename = "/".join(path_parts[1:])
    revision = _parse_revision(parsed.query)
    return repo_id, filename, revision


def _parse_hf_url(ckpt_path: str) -> Tuple[str, str, Optional[str]]:
    parsed = urlparse(ckpt_path)
    path_parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(path_parts) < 5 or path_parts[2] not in {"resolve", "blob"}:
        raise ValueError(
            "Invalid Hugging Face checkpoint URL. Use a file URL like "
            "`https://huggingface.co/<namespace>/<repo>/resolve/<revision>/<file>`."
        )

    repo_id = "/".join(path_parts[:2])
    revision = path_parts[3]
    filename = "/".join(path_parts[4:])
    return repo_id, filename, revision


def _parse_revision(query: str) -> Optional[str]:
    parsed_query = parse_qs(query)
    revisions = parsed_query.get("revision") or parsed_query.get("rev")
    if revisions:
        return revisions[0]
    return None
