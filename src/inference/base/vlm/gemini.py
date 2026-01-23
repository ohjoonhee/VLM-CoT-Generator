import os
import json
import argparse
import base64
from io import BytesIO
from typing import List, Union

import dotenv
from google import genai
from google.genai import types
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Load environment variables
dotenv.load_dotenv()

# Default configuration
# Using the model name from the user provided example
DEFAULT_MODEL_NAME = "gemini-3-flash-preview"


def _process_single_image(image_input) -> Union[Image.Image, str]:
    """
    Process a single image input into a format suitable for Gemini API (PIL Image).
    """
    if isinstance(image_input, Image.Image):
        return image_input
    elif isinstance(image_input, str):
        # Check if it's a file path
        if len(image_input) < 4096 and os.path.exists(image_input):
            return Image.open(image_input)
        return image_input  # Return as string if not a path (maybe a URL or something else, though Gemini might expect PIL)
    elif isinstance(image_input, bytes):
        return Image.open(BytesIO(image_input))
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")


def process_image(image_input) -> List[Union[Image.Image, str]]:
    """
    Process image input (single or list) into a list of PIL Images.
    """
    if isinstance(image_input, list):
        return [_process_single_image(img) for img in image_input]
    return [_process_single_image(image_input)]


def main():
    parser = argparse.ArgumentParser(description="Run Gemini inference on HF dataset.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results.")
    parser.add_argument("--dataset_name", type=str, default="ohjoonhee/Visual-CoT-4k", help="Dataset name.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (currently treating as 1 for simplicity loop).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Model name for API.")
    parser.add_argument("--image_column", type=str, default="image", help="Column name for image.")
    parser.add_argument("--question_column", type=str, default="question", help="Column name for question.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process.")

    args = parser.parse_args()

    # Initialize Gemini Client
    client = genai.Client()  # Assumes GOOGLE_API_KEY is in env

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    sanitized_model_name = args.model.replace("/", "__")
    sanitized_dataset_name = args.dataset_name.replace("/", "__")
    sanitized_split_name = args.split.replace("/", "__")

    output_file = os.path.join(args.output_dir, f"{sanitized_model_name}_{sanitized_dataset_name}_{sanitized_split_name}_results.jsonl")

    print(f"Loading dataset {args.dataset_name} split {args.split}...")
    dataset = load_dataset(args.dataset_name, split=args.split)

    if args.limit:
        dataset = dataset.select(range(args.limit))

    completed_count = 0
    # Check if output file exists to resume
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            completed_count = sum(1 for _ in f)
        print(f"Resuming from {completed_count} completed samples.")

    print(f"Starting inference with model {args.model}...")

    with open(output_file, "a", buffering=1) as f_out:
        total = len(dataset)
        for i, item in tqdm(enumerate(dataset), total=total, initial=completed_count):
            if i < completed_count:
                continue

            try:
                question = item[args.question_column]
                image_input = item[args.image_column]

                # Gemini accepts a list of [image, text, image, text...]
                # We'll construct contents as [image(s), question]

                processed_images = process_image(image_input)

                # PIL Image to bytes
                def image_to_bytes(image: Image.Image) -> bytes:
                    buffer = BytesIO()
                    image.save(buffer, format="JPEG")
                    return buffer.getvalue()

                processed_images = [
                    types.Part.from_bytes(
                        data=image_to_bytes(image),
                        mime_type="image/jpeg",
                    )
                    for image in processed_images
                ]

                contents = []
                contents.extend(processed_images)
                contents.append(question)

                response = client.models.generate_content(
                    model=args.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level="high"),
                    ),
                )

                # Parse response
                prediction = ""
                thoughts = ""

                # Handle possible multiple candidates (usually 1)
                if response.candidates:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if not part.text:
                                continue
                            if part.thought:
                                thoughts += part.text + "\n"
                            else:
                                prediction += part.text + "\n"
                            # if part.text:
                            #     if getattr(
                            #         part, "thought", False
                            #     ):  # Check if it's a thought part (SDK dependent, but user example used .thought logic, although SDK usually puts thought in metadata or specific field. Wait, user example: `if part.thought:`)
                            #         # Actually, looking at user example: `if part.thought:`
                            #         # It implies `part` object has `thought` attribute which is boolean?
                            #         # Or maybe `part.thought` is the thought text?
                            #         # User example:
                            #         # if part.thought:
                            #         #     print("Thought summary:")
                            #         #     print(part.text)

                            #         # So `part.thought` is likely a boolean flag.
                            #         if part.thought:
                            #             thoughts += part.text + "\n"
                            #         else:
                            #             prediction += part.text + "\n"

                result = {
                    "question": question,
                    "prediction": response.text.strip(),
                    # "thoughts": thoughts.strip(),
                }

                if "answer" in item:
                    result["answer"] = item["answer"]

                f_out.write(json.dumps(result) + "\n")

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                import traceback

                traceback.print_exc()
                # Optionally write error to a log file or continue


if __name__ == "__main__":
    main()
