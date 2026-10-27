# Conventions

Detailed rules for changing SMA. [AGENTS.md](AGENTS.md) is the short entry point. Rule IDs are
stable. Do not renumber them. Add new rules at the end of their section.

Each rule is one of three kinds:

- **Gate.** A tool checks it when you run that tool. Nothing runs automatically: there is no CI.
- **Review.** Nothing checks it. The author and the reviewer hold it.
- **Proposed.** Not adopted yet. It waits for an owner decision (see [Open decisions](#open-decisions)).

## Gates

- **G-1. The test suite passes.** `uv run pytest` from the root. Not run in CI. One test fails on
  machines whose hostname is the user name (see AGENTS.md).
- **G-2. Run records match their schemas.** `records.save_manifest` and `records.save_summary` scrub
  the record and validate it against `schemas/manifest.schema.json` and
  `schemas/summary.schema.json` on every write. A record change that breaks a schema fails the
  pipeline tests.
- **G-3. Config is closed.** `config.load_config` rejects unknown keys. `config.validate_config`
  rejects bad ranges and forbidden `engine.extra_args`. A new config field needs a dataclass field
  in `config.py`. A new limit needs a check in `validate_config`.
- **G-4. The corpus is pinned.** `corpus.load_corpus` refuses a corpus whose sha256 does not match.

## Product invariants

These make an SMA number trustworthy. Breaking one silently is worse than a crash. All are Review.

- **INV-1. Refuse, never convert.** SMA does not quantize, relabel, or flatten a checkpoint, and does
  not make or convert a GGUF. A mismatch is an `SmaError` that names the right flag or engine
  (`quant.assert_flag_matches`, `engines.compatibility_error`, `preflight.check_architecture`). The KV
  cache is always 8-bit (FP8, or Q8_0 on llama.cpp); no code path or `extra_args` starts a 16-bit one,
  and an engine log that reports one stops the run.
- **INV-2. Null with a reason, never a guess.** A missing measurement is `null`, and the record says
  why where it has a field for that (`measurement_note`, `missing_reason`, `invalidation_reason`,
  `scrape_error`). No zero fill. No value copied from the other engine.
- **INV-3. The engine is part of the subject.** The run path, manifest, summary, and catalog all carry
  the engine and why it was chosen (`engine.selected_by`: format, flag, default, or preflight), and
  the choice is made before the run folder exists. `sma compare` refuses a cross-engine pair unless
  `--allow-cross-engine`, and refuses llama.cpp runs, whose KL comes from llama-perplexity during the
  run. The vLLM MoE backend chain (`flashinfer_b12x`, then `marlin`) is the only fallback, and the
  backend that started is recorded.
- **INV-4. Failures are kept.** A failed step stores its scrubbed error. Invalid and unverified speed
  cells stay on disk and are left out of the headline numbers. Do not delete or hide a failed
  measurement.
- **INV-5. Inputs are pinned.** The image tag is pinned. The digest comes from
  `engine.image_digest` when the owner set it, otherwise from `docker image inspect`. It is never
  invented. The Hugging Face revision is resolved to a commit SHA. Benches run offline with
  `--tokenizer /model`.
- **INV-6. The dry-run prints what the real run executes.** Build every argv in `commands.py` or the
  engine backend. `plan.py` and `pipeline.py` both call those builders. Do not build a command
  inline in either.
- **INV-7. One GPU, one model.** Take `GpuLock` before any server starts. `run-pair` holds one lock
  for both legs. A run resumes only on the host alias that started it.
- **INV-8. Plan memory before launch.** `runplan.plan_run` sizes the context and the KV pool under
  `memory.budget_gb` from the checkpoint's own metadata, and refuses before the weight download when
  the model cannot fit. Serve commands take their context and pool from the recorded plan only.

## Records and state

- **REC-1. Write records atomically.** JSON and text in a run folder go through `sma.atomic`
  (temp file, fsync, rename; NaN and Infinity are rejected). The exceptions are `telemetry.csv` and
  `memory.csv`, append streams. *Why:* a crash must not leave a half-written manifest or summary.
- **REC-2. Steps are idempotent.** A step writes `steps/<name>.done` only after it finishes, and
  resume skips steps with a marker. A new step needs an entry in `records.STEP_NAMES`, the manifest
  schema, the README "What a run does" list, and a test that resume skips it.
- **REC-3. The record contract changes in one place at a time.** A change to the run folder or a
  record changes `schemas/`, `records.py`, and the README "Run record contract" together. Older
  folders (`runs/<slug>/<quant>/<run-id>/`) must stay readable by `list`, `status`, `resume`, and
  `compare`. A change that older readers cannot parse needs an owner decision on `schema_version`.
- **REC-4. Every resource has an owner and a teardown.** Containers, servers, threads, and locks are
  released on success, failure, Ctrl-C, and SIGTERM (`pipeline._finish` and the signal guard), and
  the outcome is recorded as a side step. A new resource names where it is released.

## Secrets, identity, and trust boundaries

- **SEC-1. Scrub before you store or log.** `log_event` scrubs its fields. Use `scrub_text` or
  `scrub_obj` for anything else you store. Use `assert_text_clean` when the value must fail closed,
  such as a path written into the record.
- **SEC-2. The token stays in the environment.** Read it with `config.hf_token`. It is never a config
  key, a CLI argument, a log field, or a record field. `sma doctor` checks that it is set and does
  not print it.
- **SEC-3. Host aliases are not hostnames.** Every alias passes `identity.assert_alias_safe`.
- **SEC-4. Everything from outside is hostile.** Hugging Face repo files, server responses, bench
  output, `nvidia-smi` output, and queue files are untrusted. Parse them defensively. Never
  unpickle: safetensors headers are read as JSON with a size cap. A bad value becomes a refusal or a
  `null` with a reason, not a crash deep in a step.
- **SEC-5. Default closed.** `trust_remote_code` is false. SMA installs no checkpoint wheel or
  plugin. The server binds to loopback. `extra_args` cannot set `--quantization`, a tensor-parallel
  size other than 1, `--host`, or `--port`.
- **SEC-6. A new runtime dependency is an owner decision.** There are five. Each is code SMA did not
  write, running in the process that holds the token. Try the standard library first: the package
  already uses `argparse` and `sqlite3`, and the test fakes use `http.server`.

## External calls and limits

- **EXT-1. Use the helpers for the outside world.** Subprocesses go through `proc.run_command` or
  `proc.run_command_streaming` (timeout, process-group kill, scrubbed output, a `what` label). HTTP
  goes through `httpapi`. Direct `subprocess` calls exist in two places, all with timeouts:
  `cleanup.py` (`sync`) and `records.git_identity` (`git`). Do not add more.
- **EXT-2. Every wait and every retry is bounded.** Waits take their limit from `cfg.timeouts`.
  Retries use `retry.call_with_retry` or a named config count (`speed.throttle_retries`,
  `queue.max_attempts`). Retry only transient failures: `httpapi` retries timeouts, 5xx, and 429,
  not other 4xx. Say what happens when the limit is reached.
- **EXT-3. A new limit is configuration.** It gets a default in `sma.yaml`, a field in `config.py`,
  a check in `validate_config`, and tests at the limit and over it. Example: `speed.num_prompts`
  above the SGLang dataset row count is refused before any download.
- **EXT-4. Expected failures are `SmaError`.** Raise `SmaError` or a subclass with a message that is
  safe to show and to store, and that tells the user what to do. The CLI prints it and exits 2. Use
  `assert` only for programmer errors, such as a context field that an earlier step must have set.

## Design

These adapt the engineering principles to SMA. All are Review.

- **DES-1. Climb the ladder after you understand the problem.** Does it need to exist? Does it
  already exist here (`atomic`, `proc`, `httpapi`, `retry`, `scrub`, `log_event`, the argv builders,
  the test fakes)? Does the standard library or an installed dependency do it? Only then write the
  minimum code. The ladder does not apply to input validation, error handling that prevents data
  loss, security, or the calibration real hardware needs.
- **DES-2. Engine differences belong in the backend.** `backends.py` says each engine owns its serve
  command, health check, logprobs, bench command, metrics names, and deterministic flags. Do not add
  new `engine == "sglang"` branches outside `backends.py` and `engines.py`. About seventeen exist
  (in `config`, `records`, `preflight`, `plan`, `cli`, `pipeline`, `queue`). When you touch one,
  consider moving it behind `EngineBackend` as its own small step. *Why:* every branch outside the
  backend is one more place that must change when an engine changes.
- **DES-3. Do not abstract early.** Two engines and four quant labels are concrete. Do not add a
  plugin registry, a generic step framework, or a base class with one user. A little duplication is
  better than an abstraction that has not earned its shape.
- **DES-4. Deep modules, simple calls.** A helper absorbs its own complexity (retries, scrubbing,
  timeouts) so callers stay short. A layer that only passes arguments through adds nothing.
- **DES-5. Readable, not clever.** Name intermediate values. If a name is hard to choose, the concept
  is probably unclear. Comments say why: a hardware finding, an owner decision, an engine quirk.
- **DES-6. File size and cohesion.** Keep handwritten files under 500 lines when practical. A file
  over 500 lines needs a cohesion review. A file over 800 lines should normally be split, by
  responsibility rather than by line count. Today `pipeline.py` is about 1,350 lines, `cli.py` about
  580, and `quant.py` about 540. Do not give `pipeline.py` a new responsibility. Put new cohesive
  logic in its own module. Splitting `pipeline.py` is its own refactor (see open decisions).
- **DES-7. Refactor small and safe.** Each step keeps the suite green. Do not mix a refactor and a
  behavior change in one commit.
- **DES-8. Design it twice.** Before a new engine, step, quant label, record field group, or queue
  protocol change, sketch two approaches and compare them on simplicity and what they hide. Put the
  comparison in the commit or PR description.
- **DES-9. Treat concurrency with caution.** Threads today: the telemetry sampler, the queue
  heartbeat, the stream readers in `proc`, and the server-log follower (`proc.follow_to_file`,
  stopped and joined in `pipeline._stop_mode`). Queue claims are an atomic `link` on one filesystem,
  and the worker takes the GPU lock before it claims. Keep jobs independent. Do not share mutable
  state between threads. A new thread needs a stop and a join in teardown.
- **DES-10. Log every major branch.** Use `log_event("<event>", run_id=..., ...)` for step changes,
  retries, fallbacks, refusals, and skipped work. Include `run_id` when one exists. The run's
  `logs/sma.log` gets every event as JSON. The terminal gets a readable line: an event the person
  running SMA should see needs a template in `logsetup._CONSOLE`. Without one it prints as
  `event key=value`.
- **DES-11. Optimize with data.** SMA's own CPU time is rarely the bottleneck. Do not optimize SMA
  code without a profile that shows the cost.
- **DES-12. Fix the root cause once.** A bug report names a symptom. Find every caller of the
  function you change and fix the shared function, not only the path in the report.
- **DES-13. Define errors out of existence only for housekeeping.** Teardown and cleanup may treat
  "already gone" as done. Measurement and identity may not: a mismatch there is a refusal, because a
  quiet success records a wrong number.

## Tests

- **TEST-1. Every bug fix has a regression test.** It fails without the fix.
- **TEST-2. Fake only at system boundaries.** `tests/fakes.py` and `tests/fake_vllm.py` stand in for
  Docker, Hugging Face, NeMo Evaluator, the bench tools, and the engine's HTTP server. Extend those fakes
  first. When you need `monkeypatch`, patch the boundary itself (`socket`, `os`, `huggingface_hub`,
  or the SMA wrapper around an outside call, such as `run_command`, `scrape_metrics`, or
  `download_snapshot`), or patch a function to prove it is never reached. Do not patch SMA's
  internal logic.
- **TEST-3. Prefer coarse tests.** Drive behavior through `pipeline.execute` with
  `tests/harness.make_config` and `make_services`, then check the written record and
  `schemautil.validate_instance`. Pure functions (KL, speed math, quant detection, Prometheus
  parsing) get direct tests.
- **TEST-4. Test the edges.** Refusals, at-limit and over-limit values, failure transitions
  (`Hooks.kill_step`, resume, a failed server start), and missing data that must become `null`.
- **TEST-5. No hardware in tests.** Tests do not need a GPU, Docker, the network, or a real image.
  When something can only be checked on hardware, add it to the README section "What this
  environment did not verify" and to `docs/first-spark-run.md`.
- **TEST-6. Test names state the behavior,** for example `test_quant_mismatch_does_not_start_vllm`.

## Docs and communication

- **DOC-1. The README is the contract.** A behavior change updates the README in the same change.
  Update `docs/arena-plan-alignment.md` when an arena-plan mapping, extension, or deviation changes.
  Update `docs/first-spark-run.md` when an operator check changes.
- **DOC-2. Say what was not verified.** Keep ran, passed, and not run apart. An engine flag or bench
  shape that never ran against the pinned image is listed as not verified, not described as working.
- **DOC-3. Speak like a human.** Short, plain sentences. Explain jargon or drop it. An error message
  says what happened and what to do next. Do not give time estimates.
- **DOC-4. Commit messages.** One summary sentence in the imperative that ends with a period (for
  example "Pin bench tokenizers to the local snapshot and reject short benches."), then a body in
  plain sentences that says what behavior changed.

## Open decisions

None of these is adopted. Each waits for the owner.

1. **No CI and no lint, format, or type gate.** The package ships `py.typed` and `.gitignore` lists
   mypy and ruff caches, but neither tool is configured. Lines reach 136 characters. Until a
   formatter exists, match the surrounding code.
2. **No lockfile.** Dependencies have lower bounds only and `uv.lock` is not committed. The
   principles say to pin versions.
3. **Runtime log-level control.** The principles ask for it. SMA logs at INFO with no switch.
4. **Splitting `pipeline.py`.** It is over the 800-line line in DES-6. A split needs a plan made of
   small, green steps.
5. **Test file names.** `test_round2.py`, `test_round3.py`, `test_review_fixes.py`, and
   `test_shannon_fixes.py` are named after review rounds. Proposed: new tests go in the file for
   the behavior they cover.
6. **Shortcut marker.** No marker exists. Proposed: mark a deliberate shortcut with a known ceiling
   with a `ponytail:` comment that names the ceiling and the upgrade path.
7. **Hostname inside a hyphenated token.** `scrub_text` replaces a hostname only when the
   characters next to it are not letters, digits, `.`, `_`, or `-`. A path such as
   `pytest-of-<host>` keeps it, which is why the pipeline test fails when hostname and user name
   match. Decide whether that is acceptable or a leak.
8. **The principles file is untracked.** `.agents/`, `.claude/skills/`, and `skills-lock.json` are
   not committed, so the GOOD_ENGINEERING link in AGENTS.md resolves only where the skill is
   installed.

## Sources

| Rules | Source |
| --- | --- |
| INV, REC, SEC-2 to SEC-5, EXT-3 | README, `docs/arena-plan-alignment.md`, and the code they cite |
| DES-1, DES-12 | The engineering skill's ladder (`.agents/skills/engineering/reference/ladder.md`) |
| DES-2 to DES-11, DES-13, TEST, DOC-3, SEC-4, SEC-6 | GOOD_ENGINEERING principles, adapted to the code above |
| DOC-4, TEST-6 | Existing commit history and test names |

Not adopted from the principles: static allocation, assertion quotas, and fixed function-length
caps (TigerStyle ideas the principles mark as not general), and the frontend guidance (SMA has no
frontend).
