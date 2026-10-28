"""A machine's own settings come from a git-ignored .env: the HF token and the cache path (owner decision 2026-10-27)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sma.config import hf_token, load_config, load_env_file
from sma.errors import SmaError

ROOT = Path(__file__).resolve().parents[1]


def test_env_file_sets_what_the_shell_has_not_and_reads_common_forms(tmp_path: Path, monkeypatch) -> None:
    # setenv first, so monkeypatch removes what load_env_file sets once the test ends.
    for key in ("HF_TOKEN", "SMA_HF_CACHE_DIR", "SMA_A", "SMA_B"):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.setenv("SMA_B", "from the shell")
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n\nHF_TOKEN=hf_example\nexport SMA_HF_CACHE_DIR='/data/models/sma/hf'\n"
        "SMA_A=plain  # a note\nSMA_B=from the file\n",
        encoding="utf-8",
    )
    assert load_env_file(env) == ["HF_TOKEN", "SMA_HF_CACHE_DIR", "SMA_A"]
    cfg = load_config(ROOT / "sma.yaml")
    assert hf_token(cfg) == "hf_example"
    assert cfg.hf.cache_dir == "/data/models/sma/hf"
    assert (os.environ["SMA_A"], os.environ["SMA_B"]) == ("plain", "from the shell")


def test_a_bad_env_line_names_the_line_and_never_its_text(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("hf_secretvalue\n", encoding="utf-8")
    with pytest.raises(SmaError, match="line 1 is not KEY=VALUE") as caught:
        load_env_file(env)
    assert "secretvalue" not in str(caught.value)


def test_an_empty_sma_env_file_reads_nothing_and_the_real_env_stays_out_of_git(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SMA_UNUSED=1\n", encoding="utf-8")
    monkeypatch.setenv("SMA_ENV_FILE", "")
    assert load_env_file() == []
    lines = {line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()}
    assert ".env" in lines
    assert "HF_TOKEN=" in (ROOT / ".env.example").read_text(encoding="utf-8")


def test_the_env_file_test_leaves_no_variables_behind() -> None:
    # Runs after the tests above in this file: nothing they loaded is still set.
    assert "SMA_HF_CACHE_DIR" not in os.environ and "SMA_A" not in os.environ


def test_lab_machine_names_from_the_env_file_become_placeholders_in_records(monkeypatch) -> None:
    from sma.scrub import scrub_obj, scrub_text

    monkeypatch.setenv("KERNELBENCH_URL", "https://kb.example-tail.ts.net/")
    monkeypatch.setenv("KERNELBENCH_HOST", "kb.example-tail.ts.net")
    monkeypatch.setenv("TAILNET_DOMAIN", "example-tail.ts.net")
    monkeypatch.setenv("SWE_RUNNER_USER", "sma")
    text = "POST https://kb.example-tail.ts.net/v1/jobs on kb.example-tail.ts.net in example-tail.ts.net as sma"
    expected = "POST ${KERNELBENCH_URL}/v1/jobs on ${KERNELBENCH_HOST} in ${TAILNET_DOMAIN} as sma"
    assert scrub_text(text) == expected
    assert scrub_obj({"argv": [text]}) == {"argv": [expected]}
