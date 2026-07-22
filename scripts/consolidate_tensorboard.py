import argparse
import json
from pathlib import Path

from monarch_distill.io import consolidate_tensorboard_scalars


def parse_args():
    parser = argparse.ArgumentParser(description="Consolidate TensorBoard scalar events into one canonical file")
    parser.add_argument("inputs", nargs="+", type=Path, help="TensorBoard event files or directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result = consolidate_tensorboard_scalars(args.inputs, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
