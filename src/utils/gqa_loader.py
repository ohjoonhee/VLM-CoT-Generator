import os
import re
import datasets
from datasets import load_dataset
from PIL import Image

IMG_ROOT = "data_viscot/cot_images_tar_split/cot_image_data/gqa"


def process_example(example):
    img_path = example["image"]
    if not img_path:
        return example
    img_path = os.path.join(IMG_ROOT, img_path)
    example["image"] = img_path

    return example


def main():
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print("CPU Count: ", cpus)
    ds = load_dataset("json", data_files="data_viscot/gqa_cot_train_filtered_heuristic.jsonl", split="train")
    # ds = ds.select(range(100))
    print(ds)
    print(ds[0])

    ds = ds.map(process_example, num_proc=max(cpus - 1, 1))

    _, ds = ds.train_test_split(test_size=2000).values()
    print(ds)

    ds = ds.cast_column("image", datasets.Image())
    print(ds[0])
    ds.push_to_hub("Visual-CoT-GQA-2k", split="train")


if __name__ == "__main__":
    main()
