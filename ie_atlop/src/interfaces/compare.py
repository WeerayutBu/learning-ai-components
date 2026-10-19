"""Compare the fixed 30-epoch, seed-66 baseline with paper Table 3."""
import argparse
import ast
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--completed", action="store_true", help="Only set after training exits successfully.")
    args = parser.parse_args()
    results = []
    for line in args.log.read_text().splitlines():
        if line.startswith("{'dev_F1':"):
            result = ast.literal_eval(line)
            results.append({key: result[key] for key in ("dev_F1", "dev_F1_ign")})
    if not results:
        raise SystemExit("No dev evaluations found; no reproduction score is available.")
    if args.completed and len(results) != 30:
        raise SystemExit("Expected 30 epoch evaluations; refusing a completed-run comparison.")
    best = max(results, key=lambda result: result['dev_F1'])
    print(json.dumps({
        "split": "dev",
        "seed": 66,
        "training_completed": args.completed,
        "evaluations": len(results),
        "best_epoch": results.index(best) + 1,
        "selection": "maximum dev_F1; Ign F1 from the same evaluation",
        "measured": best,
        "paper_five_seed_mean": {"dev_F1": 61.09, "dev_F1_ign": 59.22},
        "difference_percentage_points": {
            "dev_F1": best["dev_F1"] - 61.09,
            "dev_F1_ign": best["dev_F1_ign"] - 59.22,
        },
        "test": "unscored; requires official evaluation of artifacts/result.json",
    }, indent=2))


if __name__ == "__main__":
    main()
