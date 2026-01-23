import os
import json
import base64
import argparse
from io import BytesIO
from openai import Client
from datasets import load_dataset
from PIL import Image

# Default configuration from environment variables or defaults
BASE_URL = os.getenv("BASE_URL", "http://localhost:10630/v1")
API_KEY = os.getenv("API_KEY", "EMPTY")
MODEL_NAME = os.getenv("MODEL", "Qwen/Qwen3-VL-8B-Thinking")

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


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-VL inference on Visual-CoT dataset.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results.")
    parser.add_argument("--dataset_name", type=str, default="ohjoonhee/Visual-CoT-4k", help="Dataset name.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (currently treating as 1 for simplicity loop).")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name for API.")
    parser.add_argument("--port", type=str, default=None, help="Port override for API.")
    parser.add_argument("--image_column", type=str, default="image", help="Column name for image.")
    parser.add_argument("--question_column", type=str, default="question", help="Column name for question.")
    parser.add_argument("--system_prompt_path", type=str, default=None, help="Path to system prompt text file.")

    args = parser.parse_args()

    # Override BASE_URL if port is provided
    global BASE_URL
    if args.port:
        BASE_URL = f"http://localhost:{args.port}/v1"

    print(f"Connecting to {BASE_URL} with model {args.model}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt_path:
        print(f"Loading system prompt from {args.system_prompt_path}")
        with open(args.system_prompt_path, "r") as f:
            system_prompt = f.read()

    sanitized_model_name = args.model.replace("/", "__")
    sanitized_dataset_name = args.dataset_name.replace("/", "__")
    sanitized_split_name = args.split.replace("/", "__")

    output_file = os.path.join(args.output_dir, f"{sanitized_model_name}_{sanitized_dataset_name}_{sanitized_split_name}_results.jsonl")

    dataset = load_dataset(args.dataset_name, split=args.split)

    client = Client(base_url=BASE_URL, api_key=API_KEY)

    completed_count = 0
    # Check if output file exists to resume
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            completed_count = sum(1 for _ in f)
        print(f"Resuming from {completed_count} completed samples.")

    with open(output_file, "a", buffering=1) as f_out:
        total = len(dataset)
        for i, item in enumerate(dataset):
            if i < completed_count:
                continue

            if i % 10 == 0:
                print(f"Processing {i}/{total}")

            try:
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

                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    max_tokens=4096,
                    temperature=0.7,
                )

                prediction = response.choices[0].message.content
                # print(response)

                result = {
                    "question": question,
                    "prediction": prediction,
                    # "answer": item.get("answer", ""), # Include ground truth if available in dataset
                }
                # Check if answer exists in dataset item
                if "answer" in item:
                    result["answer"] = item["answer"]

                f_out.write(json.dumps(result) + "\n")

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Optionally write error to a log file or continue


if __name__ == "__main__":
    main()
