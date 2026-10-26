"""Upload the golden dataset to LangSmith, so the dataset there matches the JSONL file.

    python evals/upload_dataset.py [--dataset foi-golden]

Each example's id is derived from its request and received date. Re-uploading updates existing
examples instead of duplicating them, and removes any that are no longer in the file. LangSmith
keeps a version of the dataset for every change, so earlier experiments stay comparable.
"""

import argparse
import json
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

DATASET = "foi-golden"
GOLDEN = Path(__file__).parent / "datasets" / "golden.jsonl"


def example_id(example: dict) -> uuid.UUID:
    inputs = example["inputs"]
    return uuid.uuid5(uuid.NAMESPACE_URL, f"foi-golden/{inputs['received']}/{inputs['request']}")


def load_examples(path: Path = GOLDEN) -> list[dict]:
    """The JSONL examples, shaped for LangSmith, with the split as a LangSmith split."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [
        {
            "id": example_id(row),
            "inputs": row["inputs"],
            "outputs": row["outputs"],
            "metadata": row["metadata"],
            "split": row["metadata"]["split"],
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=DATASET)
    args = parser.parse_args()

    load_dotenv()
    client = Client()
    if not client.has_dataset(dataset_name=args.dataset):
        client.create_dataset(
            args.dataset, description="Golden FOI requests labelled with the ideal outcome."
        )

    examples = load_examples()
    existing = {ex.id for ex in client.list_examples(dataset_name=args.dataset)}
    wanted = {ex["id"] for ex in examples}
    new = [ex for ex in examples if ex["id"] not in existing]
    updates = [ex for ex in examples if ex["id"] in existing]
    removed = existing - wanted

    if new:
        client.create_examples(dataset_name=args.dataset, examples=new)
    if updates:
        client.update_examples(dataset_name=args.dataset, updates=updates)
    if removed:
        client.delete_examples(list(removed))
    print(f"{args.dataset}: {len(new)} added, {len(updates)} updated, {len(removed)} removed")


if __name__ == "__main__":
    main()
