import pytest

from ptlflow.utils import checkpoint_utils


class DummyModel:
    pretrained_checkpoints = {
        "things": "https://example.com/dpflow-things.ckpt",
        "hf_things": "hf://mignonjia/dpflow-ckpt/dpflow-things-2012b5d6.ckpt",
    }


def test_resolve_checkpoint_path_keeps_local_file(tmp_path) -> None:
    ckpt_path = tmp_path / "model.ckpt"
    ckpt_path.write_bytes(b"checkpoint")

    resolved_path = checkpoint_utils.resolve_checkpoint_path(str(ckpt_path))

    assert resolved_path == str(ckpt_path)


def test_resolve_checkpoint_path_downloads_hf_path(monkeypatch) -> None:
    expected_path = "/tmp/from_hf.ckpt"

    monkeypatch.setattr(
        checkpoint_utils,
        "download_hf_checkpoint",
        lambda ckpt_path: expected_path,
    )

    resolved_path = checkpoint_utils.resolve_checkpoint_path(
        "hf://mignonjia/dpflow-ckpt/dpflow-things-2012b5d6.ckpt"
    )

    assert resolved_path == expected_path


def test_resolve_checkpoint_path_downloads_hf_pretrained_alias(monkeypatch) -> None:
    expected_path = "/tmp/from_hf_alias.ckpt"

    monkeypatch.setattr(
        checkpoint_utils,
        "download_hf_checkpoint",
        lambda ckpt_path: expected_path,
    )

    resolved_path = checkpoint_utils.resolve_checkpoint_path(
        "hf_things", DummyModel
    )

    assert resolved_path == expected_path


def test_resolve_checkpoint_path_downloads_url_when_force_local(monkeypatch) -> None:
    expected_path = "/tmp/from_url.ckpt"

    monkeypatch.setattr(
        checkpoint_utils,
        "download_remote_checkpoint",
        lambda ckpt_path: expected_path,
    )

    resolved_path = checkpoint_utils.resolve_checkpoint_path(
        "things", DummyModel, force_local=True
    )

    assert resolved_path == expected_path


def test_resolve_checkpoint_path_rejects_unknown_alias() -> None:
    with pytest.raises(ValueError, match="Invalid checkpoint name unknown"):
        checkpoint_utils.resolve_checkpoint_path("unknown", DummyModel)


def test_parse_hf_checkpoint_url() -> None:
    repo_id, filename, revision = checkpoint_utils._parse_hf_checkpoint_path(
        "https://huggingface.co/mignonjia/dpflow-ckpt/resolve/main/subdir/model.ckpt"
    )

    assert repo_id == "mignonjia/dpflow-ckpt"
    assert filename == "subdir/model.ckpt"
    assert revision == "main"
