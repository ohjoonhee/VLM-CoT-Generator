import json
import datasets
from datasets import load_dataset

SOURCE_DATASET_REPO = "ohjoonhee/Visual-CoT-4k"
SOURCE_JSONL = "output/judge_filtered_reasoning_v3_v2.jsonl"


def main():
    dataset = load_dataset(SOURCE_DATASET_REPO, split="train")
    with open(SOURCE_JSONL, "r") as f:
        cot_lines = [json.loads(line) for line in f.readlines()]

    dataset = dataset.select(range(len(cot_lines)))

    def construct_messages(example, idx):
        cot_record = cot_lines[idx]
        assert cot_record["question"] == example["question"]
        assert cot_record["answer"] == example["answer"]
        example["messages"] = [
            {"role": "user", "content": "<image>" + example["question"]},
            {"role": "assistant", "content": cot_record["refined_prediction"]},
        ]
        example["cot_validation"] = cot_record["judge_result"]
        return example

    def image_to_images(example):
        img = example["image"]
        example["images"] = [img]
        return example

    dataset = dataset.map(construct_messages, with_indices=True)
    dataset = dataset.map(image_to_images)
    dataset = dataset.cast_column("images", datasets.Sequence(datasets.Image()))
    dataset = dataset.filter(lambda x: x["cot_validation"] == "1")
    dataset = dataset.remove_columns(["conversations", "question", "answer", "cot_validation", "image"])

    print(dataset)
    print(dataset[0])

    # dataset.save_to_disk("data/hf/Visual-CoT-4k-Sharegpt")
    dataset.push_to_hub("ohjoonhee/Visual-CoT-4k-Sharegpt")


if __name__ == "__main__":
    main()
