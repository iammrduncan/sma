# Config fixtures

Real `config.json` files from Hugging Face, fetched at a pinned commit. `tests/test_memplan.py` reads
them to check the memory planner against published model shapes.

| File | Repo | Commit |
| --- | --- | --- |
| `qwen3-8b.json` | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| `qwen3-30b-a3b-instruct-2507.json` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` |
| `qwen3.6-35b-a3b-nvfp4.json` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `1355db6a052410cfd62085d94b58866fd0f2c3c5` |
| `lfm2.5-8b-a1b.json` | `LiquidAI/LFM2.5-8B-A1B` | `5dd22602c2e9f6a097b1de4c4efe0658b605015c` |
| `nemotron-3.5-lightning-30b-a3b-nvfp4.json` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | `bee7596271d1495f6992ae224aefde4410e816b8` |

Each file came from `https://huggingface.co/<repo>/resolve/<commit>/config.json`.

The Qwen3, Qwen3-30B, and LFM2.5 files are byte-for-byte copies. The two NVIDIA NVFP4 files had their
`quantization_config` key removed, because it lists every quantized layer (the Nemotron one is 1.3 MB)
and the planner never reads it. Those two files were then rewritten as JSON with a 2-space indent.
Every other key and value is unchanged.

To refresh a file, look up the commit with `curl -s https://huggingface.co/api/models/<repo>` (the
`sha` field), fetch the file at that commit, and update this table.
