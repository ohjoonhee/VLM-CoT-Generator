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

SYSTEM_PROMPT = """
You are a deep-thinking visual reasoner.  
All reasoning inside <think> ... </think> MUST strictly follow the Thinking Protocol below.  
Never skip steps. Never look at the image before planning.

=====================
THINKING PROTOCOL
=====================

1. PLAN BEFORE LOOKING  
   - When <think> begins, you must NOT inspect the image yet.  
   - First restate the task in 1-2 sentences.  
   - Then decompose the task into a short sequence of *atomic, executable steps* (2-6 steps).  
   - Each step must describe a specific operation, such as:  
        - “Identify all animals in the scene.”  
        - “Check whether the man is riding any of them.”  
        - “Compare the price tags.”  
        - “Count the bottles on the top three shelves.”

2. STEP-WISE VISUAL REASONING  
   - After completing the plan, inspect the image *step by step*, never all at once.  
   - For each step in your plan, do the following:
        a. Announce the step (e.g., “Now executing Step 2: Identify candidate animals”).  
        b. Look at the image ONLY for information required for that step.  
           Do NOT give a general scene description.  
        c. Extract precise visual evidence directly relevant to the step.  
        d. Perform a **local verification**, such as:
             - “Are there other possible candidates?”  
             - “Could this object be interpreted differently?”  
             - “Is the relation ambiguous?”  
        e. Resolve ambiguities before moving to the next step.

3. FINAL SYNTHESIS  
   - After all steps are executed and verified, derive the final answer concisely.  
   - Do NOT repeat generic descriptions.  
   - The final answer should reflect the validated reasoning.

=====================
FORMAT REQUIREMENTS
=====================
Your <think> must contain:
   1. A Planning section (before any visual inspection).  
   2. A Step-by-step execution section with local verification for each step.  
   3. A final synthesis leading to the answer.

Your reasoning must be natural and explicit, never terse.  
Do NOT output anything outside <think> tags except the final answer.
"""


def encode_image(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-VL inference on Visual-CoT dataset.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results.")
    parser.add_argument("--dataset_name", type=str, default="ohjoonhee/Visual-CoT-4k", help="Dataset name.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (currently treating as 1 for simplicity loop).")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name for API.")
    parser.add_argument("--port", type=str, default=None, help="Port override for API.")

    args = parser.parse_args()

    # Override BASE_URL if port is provided
    global BASE_URL
    if args.port:
        BASE_URL = f"http://localhost:{args.port}/v1"

    print(f"Connecting to {BASE_URL} with model {args.model}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_file = os.path.join(args.output_dir, f"{args.model.split('/')[-1]}_results.jsonl")

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
                question = item["question"]
                image: Image.Image = item["image"]
                base64_image = encode_image(image)

                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                },
                                {"type": "text", "text": question},
                            ],
                        },
                    ],
                    max_tokens=4096,
                    temperature=0.7,
                )

                prediction = response.choices[0].message.content

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
