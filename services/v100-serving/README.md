# V100 serving pod

The lab's V100 pod serves models for SMA; the harness (SMA itself, every benchmark client, grading, and
records) runs on the Spark. Status: being built (2026-11-01). Today SMA does not yet start servers there; this
folder holds the pinned engine build, and the plan is in the repo owner's notes.

| | |
| --- | --- |
| Machine | A container reached over SSH (`V100_HOST` in the root `.env`): 4x V100 32 GB, driver 580, CUDA 12.8, 12 CPUs, 503 GB RAM, /work 3 TB |
| Topology | PCIe: GPUs 0+1 share one switch, 2+3 another; across the pairs traffic goes through the host bridge |
| Access | SSH from the Spark only, as `V100_HOST_USER` on `V100_HOST_SSH_PORT` with `V100_HOST_SSH_KEY`. Model ports (`V100_MODEL1_PORT`..`4`) are inside the pod; reach them with SSH local forwards |
| Engine | llama.cpp only. Volta has no BF16, FP8, or FP4, so vLLM and SGLang cannot serve here, and NVFP4 checkpoints cannot run |

## The pinned build

The pod has no Docker, and llama.cpp's published CUDA images are CUDA 13, which dropped Volta. So
`setup.sh` builds llama.cpp from source at release b11960 (the Spark's pinned release) for compute capability
7.0 with the pod's CUDA 12.8 toolkit:

```bash
scp -P "$V100_HOST_SSH_PORT" -i "$V100_HOST_SSH_KEY" services/v100-serving/setup.sh "$V100_HOST_USER@$V100_HOST:/work/sma/services/"
ssh -p "$V100_HOST_SSH_PORT" -i "$V100_HOST_SSH_KEY" "$V100_HOST_USER@$V100_HOST" 'bash /work/sma/services/setup.sh /work/sma'
```

It writes `/work/sma/llama.cpp-b11960/build/bin/llama-server` and `llama-perplexity`, and `build.json` with the
tag, the commit, the CUDA version, the build flags, and each binary's sha256. A run's record names that build
the way a Spark run names its image digest.
