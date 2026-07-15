import argparse
import os
import sys

from monarch_distill import MonarchCompressor, default_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Distill Gemma MLPs into Monarch factors")
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Cumulative unfrozen_weights.pt checkpoint from the last completed module",
    )
    parser.add_argument(
        "--resume-start-module-index",
        type=int,
        help="Zero-based index of the first unfinished module",
    )
    args = parser.parse_args(argv)
    if (args.resume_from_checkpoint is None) != (args.resume_start_module_index is None):
        parser.error("resume checkpoint and resume module index must be provided together")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = default_config()
    if args.resume_from_checkpoint is not None:
        config.resume_from_checkpoint = args.resume_from_checkpoint
        config.resume_start_module_index = args.resume_start_module_index

    compressor = MonarchCompressor(config)
    success = False
    try:
        compressor.run_compression()
        success = True
    finally:
        compressor.close()

    if success and config.force_exit_after_success:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
