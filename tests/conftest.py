"""Keep the developer's own .env (HF token, cache path) out of the tests, in-process and in subprocesses."""

import os

os.environ["SMA_ENV_FILE"] = ""
