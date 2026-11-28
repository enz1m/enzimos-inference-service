# SPDX-License-Identifier: MIT
"""Extract the unmodified ONNX graphs and vocabulary from the English bundle."""
import argparse
import json
from pathlib import Path
import struct


def field(source, limit):
    prefix = source.read(4)
    if len(prefix) != 4:
        raise ValueError("Missing Kroko field length")
    size, = struct.unpack("<I", prefix)
    if not 0 < size <= limit:
        raise ValueError("Invalid Kroko field length")
    data = source.read(size)
    if len(data) != size:
        raise ValueError("Truncated Kroko field")
    return data


def unpack(bundle, destination):
    with bundle.open("rb") as source:
        header = json.loads(field(source, 4096))
        if (header.get("id") != "2e232c7e-a977-400e-ac93-0918a77ac9a4"
                or header.get("free") is not True or header.get("type") != "zipformer2"
                or header.get("language", {}).get("iso") != "EN"):
            raise ValueError("Unexpected Kroko model")
        destination.mkdir(parents=True, exist_ok=True)
        for name, limit in (("encoder.onnx", 160_000_000), ("decoder.onnx", 1_000_000),
                            ("joiner.onnx", 1_000_000), ("tokens.txt", 100_000)):
            (destination / name).write_bytes(field(source, limit))
        if source.read(1):
            raise ValueError("Unexpected trailer in Kroko model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    unpack(args.bundle, args.destination)
