#!/usr/bin/env python3
"""Build a parallel CSV from Evacun-style aligned text files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Convert aligned Evacun txt files into a train.csv file.")
    parser.add_argument("--transcription_path", required=True, help="Path to Akkadian transcription txt file.")
    parser.add_argument("--english_path", required=True, help="Path to English translation txt file.")
    parser.add_argument("--output_path", required=True, help="Path to output CSV file.")
    parser.add_argument("--dataset_name", default="evacun", help="Dataset name to store in the CSV.")
    parser.add_argument("--id_prefix", default="evacun", help="Prefix used when generating example ids.")
    return parser.parse_args()


def read_lines(path: str) -> list[str]:
    """Read non-empty lines from a text file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input text file not found: {path}")
    with file_path.open("r", encoding="utf-8") as fp:
        return [line.rstrip("\n") for line in fp]


def main() -> None:
    """Build a CSV from aligned transcription/translation text files."""
    args = parse_args()
    source_lines = read_lines(args.transcription_path)
    target_lines = read_lines(args.english_path)

    if len(source_lines) != len(target_lines):
        raise ValueError(
            "Input text files must have the same number of lines: "
            f"{args.transcription_path} ({len(source_lines)}) vs {args.english_path} ({len(target_lines)})"
        )

    rows = []
    for index, (source, target) in enumerate(zip(source_lines, target_lines)):
        source = source.strip()
        target = target.strip()
        if not source or not target:
            continue
        rows.append(
            {
                "id": f"{args.id_prefix}-{index:06d}",
                "source": source,
                "target": target,
                "dataset": args.dataset_name,
            }
        )

    if not rows:
        raise ValueError("No aligned non-empty sentence pairs were found in the provided files.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
