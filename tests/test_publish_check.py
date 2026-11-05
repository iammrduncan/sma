"""sma publish-check: nothing private reaches the public repository (audit of 2026-11-02)."""

from __future__ import annotations

from pathlib import Path

from sma import cli, scrub


def _private(monkeypatch) -> None:
    monkeypatch.setattr(scrub, "host_names", lambda: ["spark-secret"])
    monkeypatch.setenv("V100_HOST", "203.0.113.7")
    monkeypatch.setenv("SMA_PUBLISH_DENY", "secret-lab, x")  # a name under 3 characters is ignored


def test_publish_check_names_what_would_leak_and_never_quotes_it(tmp_path: Path, monkeypatch, capsys) -> None:
    _private(monkeypatch)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "notes.md").write_text("Written by Secret-Lab on spark-secret.\n", encoding="utf-8")
    (tmp_path / "key.txt").write_text("token hf_" + "a" * 34 + "\n", encoding="utf-8")
    (tmp_path / "clean.md").write_text("Served on a V100 slot; x marks nothing; tokens per second.\n", encoding="utf-8")

    assert cli.main(["publish-check", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "notes.md: this machine's hostname, a name in SMA_PUBLISH_DENY" in out
    assert "key.txt: a Hugging Face token" in out
    assert "clean.md" not in out
    assert "2 of 3 files would publish" in out
    assert "spark-secret" not in out and "hf_aaa" not in out and "Secret-Lab" not in out


def test_fix_scrubs_run_files_again_but_leaves_documents_and_binaries(tmp_path: Path, monkeypatch, capsys) -> None:
    # The V100 runs stored the pod's address before V100_HOST was a lab name; --fix writes what SMA writes now.
    _private(monkeypatch)
    run = tmp_path / "runs" / "Org_Model" / "llamacpp-v100" / "gguf-F16" / "run-1"
    (run / "logprobs").mkdir(parents=True)
    command = run / "logprobs" / "command.json"
    command.write_text('{"argv": ["ssh", "sma@203.0.113.7", "llama-perplexity"]}\n', encoding="utf-8")
    binary = run / "cache.db"
    binary.write_bytes(b"\xff\xfe203.0.113.7\x00")

    assert cli.main(["publish-check", "--fix", str(run)]) == 1  # the binary cannot be rewritten safely
    out = capsys.readouterr().out
    assert command.read_text(encoding="utf-8") == '{"argv": ["ssh", "sma@${V100_HOST}", "llama-perplexity"]}\n'
    assert f"scrubbed {command}" in out
    assert f"{binary}: V100_HOST" in out and binary.read_bytes() == b"\xff\xfe203.0.113.7\x00"

    doc = tmp_path / "README.md"
    doc.write_text("The pod is 203.0.113.7.\n", encoding="utf-8")
    assert cli.main(["publish-check", "--fix", str(doc)]) == 1
    assert doc.read_text(encoding="utf-8") == "The pod is 203.0.113.7.\n"

    binary.unlink()
    assert cli.main(["publish-check", str(run)]) == 0
    assert "Nothing to hide in 1 files." in capsys.readouterr().out


def test_in_the_repository_only_what_git_publishes_is_read(tmp_path: Path, monkeypatch, capsys) -> None:
    # A V100 run keeps a 2 GB kld-base.bin that git ignores and the manifest hashes: the check once read it whole.
    from sma.proc import require_ok, run_command

    _private(monkeypatch)
    require_ok(run_command(["git", "init", "-q", str(tmp_path)], timeout_s=30, what="git init"), what="git init")
    (tmp_path / ".gitignore").write_text("kld-base.bin\n", encoding="utf-8")
    run = tmp_path / "runs" / "run-1"
    run.mkdir(parents=True)
    (run / "kld-base.bin").write_text("sma@203.0.113.7\n", encoding="utf-8")
    (run / "manifest.json").write_text('{"host_alias": "gpu-host-1"}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert cli.main(["publish-check", "--fix", "runs/run-1"]) == 0
    assert "Nothing to hide in 1 files." in capsys.readouterr().out
    assert (run / "kld-base.bin").read_text(encoding="utf-8") == "sma@203.0.113.7\n"


def test_key_shapes_are_found_whatever_their_source() -> None:
    # Built here, so this file does not trip the check it tests.
    assert scrub.find_leaks("-----BEGIN " + "OPENSSH PRIVATE KEY-----") == ["a private key"]
    assert scrub.find_leaks("ghp_" + "b" * 36) == ["a GitHub token"]
    assert scrub.find_leaks("sk-or-" + "v1-" + "0a" * 32) == ["an OpenRouter key"]
    assert scrub.find_leaks('{"Authorization": "Bearer your_api_key_here", "hf_token": null}') == []
