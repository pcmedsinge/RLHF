"""
Data Filter Script — Phase 2
=============================
Purpose: Filter Phase 1 dataset to only rows with review_status=approved.
These are the high-confidence preference pairs used for DPO training.

Analogy: Before teaching the resident, the senior consultant reviews all
practice cases and stamps only the verified ones as "approved for use."
We train only on those stamped cases.
"""
import json
from pathlib import Path


def filter_approved(input_path: Path, output_path: Path) -> int:
    approved = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("metadata", {}).get("review_status") == "approved":
                approved.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in approved:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return len(approved)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"

    pairs = [
        (processed / "phase1_train.jsonl", processed / "phase2_train_approved.jsonl"),
        (processed / "phase1_eval.jsonl",  processed / "phase2_eval_approved.jsonl"),
    ]

    for src, dst in pairs:
        count = filter_approved(src, dst)
        total = sum(1 for _ in src.open("r", encoding="utf-8"))
        split = "train" if "train" in src.name else "eval"
        print(f"[{split}] {count}/{total} rows approved → {dst.name}")


if __name__ == "__main__":
    main()
