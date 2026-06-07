import os
import sys

from monarch_distill import MonarchCompressor, default_config


def main():
    config = default_config()
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
