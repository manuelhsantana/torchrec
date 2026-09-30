#!/usr/bin/env python3
"""Patch fbgemm_gpu-cpu (1.7.0) to add a `ComputeDevice.XPU` enum member.

torchrec's BatchedFusedEmbedding/BatchedFusedEmbeddingBag reference
`ComputeDevice.XPU` directly (see
torchrec/distributed/batched_embedding_kernel.py), but upstream
fbgemm_gpu-cpu's ComputeDevice enum only defines CPU/CUDA/MTIA, so that
attribute access raises AttributeError before any table can be built on XPU.

ComputeDevice is canonically defined in fbgemm_gpu/tbe/config/embedding_config.py
and re-exported unchanged through split_table_batched_embeddings_ops_common.py,
so patching it there is sufficient for every importer. Idempotent — safe to
re-run.
"""
import os
import sys

import fbgemm_gpu

FBGEMM_DIR = os.path.dirname(fbgemm_gpu.__file__)
TARGET = os.path.join(FBGEMM_DIR, "tbe", "config", "embedding_config.py")

REPLACEMENTS = [
    (
        "ComputeDevice enum members",
        (
            "class ComputeDevice(enum.IntEnum):\n"
            "    CPU = 0\n"
            "    CUDA = 1\n"
            "    MTIA = 2\n"
        ),
        (
            "class ComputeDevice(enum.IntEnum):\n"
            "    CPU = 0\n"
            "    CUDA = 1\n"
            "    MTIA = 2\n"
            "    XPU = 3\n"
        ),
    ),
]


def main() -> int:
    with open(TARGET, "r") as f:
        content = f.read()

    if "XPU = 3" in content:
        print(f"Already patched: {TARGET}")
        return 0

    changed = 0
    for label, old, new in REPLACEMENTS:
        if old not in content:
            print(f"ERROR: expected block not found for {label} — fbgemm_gpu-cpu"
                  " version mismatch? Aborting without partial patch.", file=sys.stderr)
            return 1
        content = content.replace(old, new, 1)
        changed += 1

    with open(TARGET, "w") as f:
        f.write(content)
    print(f"Patched {changed} block(s) in {TARGET}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
