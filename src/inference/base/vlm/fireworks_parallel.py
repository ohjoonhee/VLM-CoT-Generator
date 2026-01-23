import dotenv
import os
import json
import base64
import argparse
import requests
from io import BytesIO
import concurrent.futures
from functools import partial
from openai import Client
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

dotenv.load_dotenv()

# Default configuration from environment variables or defaults
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1/chat/completions")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "EMPTY")
FIREWORKS_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/qwen3-vl-235b-a22b-thinking")


DEFAULT_SYSTEM_PROMPT = ""


def _process_single_image(image_input):
    if isinstance(image_input, Image.Image):
        buffered = BytesIO()
        image_input.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    elif isinstance(image_input, str):
        # Check if it's a file path
        if len(image_input) < 4096 and os.path.exists(image_input):
            image = Image.open(image_input)
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        return image_input
    elif isinstance(image_input, bytes):
        return base64.b64encode(image_input).decode("utf-8")
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")


def process_image(image_input):
    if isinstance(image_input, list):
        return [_process_single_image(img) for img in image_input]
    return [_process_single_image(image_input)]


def process_sample(idx, dataset, args, system_prompt, base_url, api_key):
    try:
        item = dataset[idx]
        question = item[args.question_column]
        image_input = item[args.image_column]
        base64_images = process_image(image_input)

        content = []
        for b64_img in base64_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                }
            )
        content.append({"type": "text", "text": question})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        payload = {
            "model": args.model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stream": False,
        }
        headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}

        response = requests.post(base_url, json=payload, headers=headers)

        # Check if request was successful before parsing
        if response.status_code != 200:
            return None, f"HTTP Error {response.status_code}: {response.text}"

        response_json = response.json()
        if "choices" not in response_json or not response_json["choices"]:
            return None, f"Invalid API response: {response.text}"

        prediction = response_json["choices"][0]["message"]["content"]

        result = {
            "question_id": item["question_id"],
            "question": question,
            "prediction": prediction,
        }
        # Check if answer exists in dataset item
        if "answer" in item:
            result["answer"] = item["answer"]

        return result, None

    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-VL inference on Visual-CoT dataset (Multi-threaded).")
    parser.add_argument("--output_dir", type=str, default="output/multithreaded/", help="Directory to save results.")
    parser.add_argument("--dataset_name", type=str, default="ohjoonhee/Visual-CoT-60k", help="Dataset name.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (unused in MT version, kept for compatibility).")
    parser.add_argument("--model", type=str, default=FIREWORKS_MODEL, help="Model name for API.")
    parser.add_argument("--port", type=str, default=None, help="Port override for API.")
    parser.add_argument("--image_column", type=str, default="image", help="Column name for image.")
    parser.add_argument("--question_column", type=str, default="question", help="Column name for question.")
    parser.add_argument("--system_prompt_path", type=str, default="configs/prompts/think_first_v5.txt", help="Path to system prompt text file.")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of concurrent workers.")

    args = parser.parse_args()

    # Override BASE_URL if port is provided
    base_url = FIREWORKS_BASE_URL
    if args.port:
        base_url = f"http://localhost:{args.port}/v1"

    print(f"Connecting to {base_url} with model {args.model}")
    print(f"Using {args.num_workers} workers.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt_path:
        print(f"Loading system prompt from {args.system_prompt_path}")
        if os.path.exists(args.system_prompt_path):
            with open(args.system_prompt_path, "r") as f:
                system_prompt = f.read()
        else:
            print(f"Warning: System prompt path {args.system_prompt_path} does not exist. Using default.")

    sanitized_model_name = args.model.replace("/", "__")
    sanitized_dataset_name = args.dataset_name.replace("/", "__")
    sanitized_split_name = args.split.replace("/", "__")

    output_file = os.path.join(args.output_dir, f"{sanitized_model_name}_{sanitized_dataset_name}_{sanitized_split_name}_results.jsonl")

    dataset = load_dataset(args.dataset_name, split=args.split)

    completed_count = 0
    # Check if output file exists to resume
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            completed_count = sum(1 for _ in f)
        print(f"Resuming from {completed_count} completed samples.")

    # Slice dataset to skip processed items
    # Note: If previous run failed on an item and didn't write it, we retry it.
    # dataset.select expects indices, but slicing a dataset object directly is often supported or we use select.
    # Safe way with datasets library:
    if completed_count < len(dataset):
        dataset_to_process = dataset.select(range(completed_count, len(dataset)))
    else:
        dataset_to_process = []
        print("All items processed.")
        return

    with open(output_file, "a", buffering=1) as f_out:
        total = len(dataset)

        # Use partial to bind constant arguments
        worker_func = partial(process_sample, dataset=dataset_to_process, args=args, system_prompt=system_prompt, base_url=base_url, api_key=FIREWORKS_API_KEY)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # executor.map maintains order
            # Pass indices instead of items to avoid loading all images into memory
            results = executor.map(worker_func, range(len(dataset_to_process)))

            success_count = 0
            error_count = 0

            with tqdm(total=total, initial=completed_count, unit="sample") as pbar:
                for i, (result, error) in enumerate(results):
                    current_idx = completed_count + i

                    if error:
                        error_count += 1
                        pbar.write(f"Error processing sample {current_idx}: {error}")
                        # Consistent with original: do not write failed samples.
                        # This means on restart we will retry this sample.
                        continue

                    success_count += 1
                    if result:
                        f_out.write(json.dumps(result) + "\n")

                    pbar.set_postfix({"success": success_count, "error": error_count})
                    pbar.update(1)


if __name__ == "__main__":
    main()
