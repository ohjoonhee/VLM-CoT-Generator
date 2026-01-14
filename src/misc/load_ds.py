import os
import re
import datasets
from datasets import load_dataset
from PIL import Image

IMG_ROOT = "data_viscot/cot_images_tar_split/cot_image_data"


def process_example(example):
    img_paths = example["image"]
    if not img_paths:
        return example
    imgs = []
    for i, img_path in enumerate(img_paths):
        if "###" in img_path:
            continue
            # img_path, bbox = img_path.split("###")
            # img_path = os.path.join(IMG_ROOT, img_path.replace("cot/", ""))
            # bbox = [int(x) for x in eval(bbox.strip())]
            # imgs.append(Image.open(img_path).convert("RGB").crop(bbox))
        else:
            img_path = os.path.join(IMG_ROOT, img_path.replace("cot/", ""))
            imgs.append(Image.open(img_path).convert("RGB"))
    assert len(imgs) == 1
    example["image"] = imgs[0]

    conv = example["conversations"]
    pattern = r"<image>\s*(.*?)\s*Please provide the bounding"

    assert conv[0]["from"] == "human"
    questions = re.findall(pattern, conv[0]["value"], flags=re.DOTALL)

    if not questions:
        raise ValueError("No question parsed")
    prompt = questions[0].strip()
    # print(prompt)

    assert conv[-1]["from"] == "gpt"
    completion = conv[-1]["value"]
    # print(completion)

    example["question"] = prompt
    example["answer"] = completion

    return example


def main():
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print("CPU Count: ", cpus)
    ds = load_dataset("json", data_files="data_viscot/viscot_363k.json", split="train")
    # ds = ds.select(range(100))
    print(ds)
    print(ds[0])

    ds: datasets.Dataset = ds.cast_column("dataset", datasets.ClassLabel(names=ds.unique("dataset")))
    _, ds = ds.train_test_split(test_size=60_000, stratify_by_column="dataset").values()
    print(ds)

    ds = ds.map(process_example, num_proc=max(cpus - 1, 1))
    ds = ds.cast_column("image", datasets.Image())
    print(ds[0])
    ds.push_to_hub("Visual-CoT-60k", split="train")


if __name__ == "__main__":
    main()
