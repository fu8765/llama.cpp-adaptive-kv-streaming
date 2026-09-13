#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf

logger = logging.getLogger("gguf-condense-dflash")

# the only tensors whose shape depends on the vocabulary
SELECTOR_TENSORS = (
    "selector_predecessor.weight",
    "selector_successor.weight",
)


def token_list(reader: gguf.GGUFReader) -> list[bytes]:
    field = reader.get_field("tokenizer.ggml.tokens")
    return [bytes(field.data[i]) for i in range(len(field.data))]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Condense a DFlash draft model onto the vocabulary of a pre-condensed target model")
    parser.add_argument("draft", type=Path, help="DFlash draft GGUF with the full target vocabulary")
    parser.add_argument("target", type=Path, help="condensed target model GGUF")
    parser.add_argument("output", type=Path, help="output GGUF")
    args = parser.parse_args()

    logger.info("Loading %s", args.draft)
    d = gguf.GGUFReader(args.draft, "r")
    logger.info("Loading %s", args.target)
    t = gguf.GGUFReader(args.target, "r")

    arch = d.get_field(gguf.Keys.General.ARCHITECTURE).contents()

    dt = token_list(d)
    ct = token_list(t)
    logger.info("draft tokens %d, target tokens %d", len(dt), len(ct))

    # the condensed vocabulary is an ordered subsequence of the draft vocabulary
    mp: list[int] = []
    i = 0
    for token in ct:
        while i < len(dt) and dt[i] != token:
            i += 1
        if i >= len(dt):
            logger.error("target tokenizer is not a subsequence of the draft tokenizer")
            sys.exit(1)
        mp.append(i)
        i += 1
    logger.info("mapped %d of %d tokens", len(mp), len(ct))

    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=d.endianess)
    if d.get_field(gguf.Keys.General.ALIGNMENT) is not None:
        writer.data_alignment = d.get_field(gguf.Keys.General.ALIGNMENT).contents()

    for field in d.fields.values():
        if field.name in (gguf.Keys.General.ARCHITECTURE, gguf.Keys.General.ALIGNMENT, "dflash.mask_token_id"):
            continue
        if field.name.startswith("tokenizer.ggml.") or field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)

    for field in t.fields.values():
        if not field.name.startswith("tokenizer.ggml.") or field.name == "tokenizer.ggml.mask_token_id":
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)

    # the noise-fill mask must exist in the condensed vocabulary; the special
    # tokens keep their string, so map the draft mask by string
    dm = d.get_field("dflash.mask_token_id")
    dtm = d.get_field("tokenizer.ggml.mask_token_id")
    old = int((dtm if dtm is not None else dm).contents())
    d_strs = list(d.get_field("tokenizer.ggml.tokens").contents())
    t_strs = list(t.get_field("tokenizer.ggml.tokens").contents())
    if d_strs[old] in t_strs:
        new = t_strs.index(d_strs[old])
    else:
        new = int(t.get_field("tokenizer.ggml.eos_token_id").contents())
    logger.info("mask token %d (%s) -> %d", old, d_strs[old], new)
    writer.add_uint32("tokenizer.ggml.mask_token_id", new)
    writer.add_uint32("dflash.mask_token_id", new)

    mp_arr = np.asarray(mp, dtype=np.int64)
    sliced: dict[str, np.ndarray] = {}
    for tensor in d.tensors:
        if tensor.name in SELECTOR_TENSORS:
            sel = np.ascontiguousarray(np.asarray(tensor.data)[mp_arr, :])
            sliced[tensor.name] = sel
            writer.add_tensor_info(tensor.name, sel.shape, sel.dtype, sel.nbytes, tensor.tensor_type)
        else:
            writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for tensor in d.tensors:
        data = sliced.get(tensor.name, tensor.data)
        writer.write_tensor_data(data, tensor_endianess=d.endianess)

    writer.close()
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
