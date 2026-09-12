#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf

logger = logging.getLogger("gguf-extract-mtp")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the MTP/nextn block of a merged GGUF into a standalone MTP-only GGUF. "
            "The output keeps the target vocab metadata and the MTP block, so the model loader "
            "detects mtp_only and skips the trunk tensors."
        )
    )
    parser.add_argument("input", type=Path, help="merged target GGUF containing the MTP block")
    parser.add_argument("output", type=Path, help="MTP-only GGUF to write")
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        metavar="PREFIX",
        help="extra tensor name prefix to keep (repeatable)",
    )
    parser.add_argument(
        "--with-lm-head",
        action="store_true",
        help="keep output.weight instead of borrowing the target LM head",
    )
    args = parser.parse_args()

    reader = gguf.GGUFReader(args.input, "r")

    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        raise ValueError(f"{args.input}: missing general.architecture")
    arch = arch_field.contents()

    block_count_field = reader.get_field(f"{arch}.block_count")
    if block_count_field is None:
        raise ValueError(f"{args.input}: missing {arch}.block_count")
    block_count = block_count_field.contents()

    nextn_field = reader.get_field(f"{arch}.nextn_predict_layers")
    nextn = nextn_field.contents() if nextn_field is not None else 0
    if nextn == 0:
        raise ValueError(f"{args.input}: {arch}.nextn_predict_layers is 0, no MTP block to extract")

    mtp_idx = block_count - nextn
    logger.info("arch=%s block_count=%s nextn=%s mtp_idx=%s", arch, block_count, nextn, mtp_idx)

    keep_names = {"token_embd.weight", "output_norm.weight"}
    if args.with_lm_head:
        keep_names.add("output.weight")
    keep_prefixes = [f"blk.{mtp_idx}." , *args.keep]

    tensors = [
        t for t in reader.tensors
        if t.name in keep_names or any(t.name.startswith(prefix) for prefix in keep_prefixes)
    ]
    if not tensors:
        raise ValueError(f"{args.input}: no MTP tensors found for block {mtp_idx}")

    total = sum(t.n_bytes for t in tensors)
    logger.info("keeping %d tensors (%.1f MiB)", len(tensors), total / 1024 / 1024)
    for t in tensors:
        logger.info("  %s %s", t.name, list(t.data.shape))

    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=reader.endianess)

    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)

    for t in tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for t in tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)

    writer.close()
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
