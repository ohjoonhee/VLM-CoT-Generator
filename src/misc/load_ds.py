import os
import datasets
from datasets import load_dataset
from PIL import Image

IMG_ROOT = "data_viscot/cot_images_tar_split/cot_image_data"


def process_image(example):
    img_paths = example["image"]
    if not img_paths:
        return example
    imgs = []
    for i, img_path in enumerate(img_paths):
        if "###" in img_path:
            img_path, bbox = img_path.split("###")
            img_path = os.path.join(IMG_ROOT, img_path.replace("cot/", ""))
            bbox = [int(x) for x in eval(bbox.strip())]
            imgs.append(Image.open(img_path).convert("RGB").crop(bbox))
        else:
            img_path = os.path.join(IMG_ROOT, img_path.replace("cot/", ""))
            imgs.append(Image.open(img_path).convert("RGB"))
    example["image"] = imgs
    return example


def main():
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print("CPU Count: ", cpus)
    ds = load_dataset("json", data_files="data_viscot/viscot_363k.json", split="train")
    # ds = ds.select(range(100))
    print(ds)
    print(ds[0])

    ds: datasets.Dataset = ds.cast_column("dataset", datasets.ClassLabel(names=ds.unique("dataset")))
    _, ds = ds.train_test_split(test_size=0.1, stratify_by_column="dataset").values()
    print(ds)
    ds = ds.cast_column("dataset", datasets.Value("string"))

    ds = ds.map(process_image, num_proc=max(os.cpu_count() - 1, 1))
    ds = ds.cast_column("image", datasets.List(datasets.Image()))
    print(ds[0])
    ds.push_to_hub("Visual-CoT-Sampled", split="train", num_proc=max(os.cpu_count() - 1, 1))


if __name__ == "__main__":
    main()
