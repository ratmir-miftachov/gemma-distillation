from __future__ import annotations

import argparse
from pathlib import Path

from tinyhellaswag_benchmark import (
    DEFAULT_SEED,
    compare_results,
    load_result,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two TinyHellaSwag result.json files."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, default=Path("comparison.json"))
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = compare_results(
        load_result(args.baseline),
        load_result(args.candidate),
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    write_json(args.output, comparison)
    gpirt = comparison["official_gp_irt"]
    raw = comparison["raw_anchor_accuracy"]
    mcnemar = comparison["mcnemar_exact"]
    print(
        f"GP-IRT delta: {gpirt['delta']:+.4f}\n"
        f"Raw accuracy delta: {raw['delta']:+.4f} "
        f"(95% CI {raw['paired_bootstrap_95_percent_ci']})\n"
        f"Disagreements: {comparison['disagreement_count']}\n"
        f"McNemar exact p: {mcnemar['p_value']:.6f}\n"
        f"Comparison: {args.output}"
    )


if __name__ == "__main__":
    main()
