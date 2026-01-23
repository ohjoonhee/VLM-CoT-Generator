import json
import os
import re
from openai import OpenAI
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()

# Files
import sys

if len(sys.argv) > 1:
    INPUT_FILE = sys.argv[1]
else:
    INPUT_FILE = "output/multithreaded/accounts__fireworks__models__qwen3-vl-235b-a22b-thinking_ohjoonhee__Visual-CoT-60k_train_results.jsonl"
OUTPUT_FILE = INPUT_FILE.replace(".jsonl", "_judged.jsonl")

# Model configuration
MODEL = "gpt-4.1-nano-2025-04-14"
API_KEY = os.getenv("OPENAI_API_KEY", None)

BINARY_JUDGE_PROMPT = """You are a strict evaluator assessing answer correctness. You must output {positive} for fully correct answers and {negative} for any other case.

# Input
Question:
```
{question}
```
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{prediction}
```

# Evaluation Rules
- The model prediction may contain the reasoning process, you should spot the final answer from it.
- Score {positive} if the prediction matches the answer semantically, even if the format differs.
- Score {positive} if the prediction includes the correct answer along with additional context.
- Score {negative} if the prediction contradicts or fails to include the correct answer.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Strict Output format
{positive} or {negative}"""


def parse_prediction(prediction):
    """
    Parses the prediction to strip off the reasoning <think> ... </think>
    and leave only the content after </think>.
    """
    if "</think>" in prediction:
        return prediction.split("</think>")[-1].strip()
    return prediction.strip()


import concurrent.futures
import threading

# ... (other imports remain, but will be re-added in full replacement to ensure consistency)

MAX_WORKERS = 16


def process_line(line, client_kwargs):
    """
    Process a single line of JSONL.
    client_kwargs: dict containing api_key, etc. to instantiate client inside thread
                   OR pass a shared client if it is thread-safe (OpenAI client is thread-safe).
    """
    # It's better to instantiate client once per thread or use a shared one.
    # The official OpenAI Python client is thread-safe.
    # We will pass the client instance or use a global one if we want.
    # To be safe and clean, let's just use a global client or pass it.
    # Actually, simpler to just do logic here.

    line = line.strip()
    if not line:
        return None

    try:
        record = json.loads(line)
        prediction = record.get("prediction", "")

        # parser prediction
        final_answer = parse_prediction(prediction)
        record["parsed_prediction"] = final_answer

        if not final_answer:
            # If empty prediction/answer, answer is wrong
            record["judge_result"] = "0"
            return record

        # Prepare the prompt
        messages = [
            {
                "role": "user",
                "content": BINARY_JUDGE_PROMPT.format(question=record.get("question", ""), answer=record.get("answer", ""), prediction=final_answer, positive="1", negative="0"),
            },
        ]

        # Instantiate client here or use shared one?
        # OpenAI client is thread-safe, so we can use a shared one.
        # But we need to pass it in.
        # Let's instantiate a local client if needed, or just use the global API_KEY.
        # Since we are inside a function, let's create a partial or just use a global helper if we want.
        # However, to avoid passing client around, we can instantiate inside if it's cheap,
        # but connection pooling might be better with a shared client.

        client = OpenAI(**client_kwargs)  # create new client per thread to be safe/simple or rely on connection pooling

        try:
            response = client.chat.completions.create(model=MODEL, messages=messages)
            judge_result = response.choices[0].message.content.strip()
            # Ensure we get 0 or 1
            if "1" in judge_result:
                record["judge_result"] = "1"
            elif "0" in judge_result:
                record["judge_result"] = "0"
            else:
                # Fallback
                record["judge_result"] = judge_result

        except Exception as e:
            # print(f"API call failed: {e}")
            # Printing might race, maybe just store error
            record["error"] = str(e)

        return record

    except json.JSONDecodeError:
        return None
    except Exception as e:
        return None  # or return record with error


def judge_answers():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Reading from {INPUT_FILE}...", flush=True)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Writing to {OUTPUT_FILE}...", flush=True)
    print(f"Processing {len(lines)} items with {MAX_WORKERS} threads...", flush=True)

    client_kwargs = {"api_key": API_KEY}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Create a future for each line
            # We use a list comprehension or generator
            futures = [executor.submit(process_line, line, client_kwargs) for line in lines]

            # Use tqdm to monitor progress
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(lines), desc="Processing items"):
                try:
                    result = future.result()
                    if result:
                        outfile.write(json.dumps(result) + "\n")
                        outfile.flush()
                except Exception as e:
                    print(f"Error processing future: {e}", flush=True)


if __name__ == "__main__":
    judge_answers()
