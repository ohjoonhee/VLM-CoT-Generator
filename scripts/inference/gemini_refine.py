import json
import os
from openai import OpenAI
from google import genai
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()

# Files
INPUT_FILE = "output/slurm/Qwen__Qwen3-VL-235B-A22B-Thinking-FP8_ohjoonhee__Visual-CoT-4k_train_results.jsonl"
OUTPUT_FILE = "output/gemini_refined_reasoning_v3.jsonl"
REFINE_PROMPT_PATH = "configs/prompts/llm_refine_v3.txt"
with open(REFINE_PROMPT_PATH, "r", encoding="utf-8") as f:
    REFINE_PROMPT = f.read()


# Model configuration
MODEL = "gemini-3-flash-preview"
# API_KEY = os.getenv("OPENAI_API_KEY", None)


def refine_jsonl():
    client = genai.Client()

    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print(f"Reading from {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in tqdm(infile, desc="Processing lines"):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                prediction = record.get("prediction", "")

                if not prediction:
                    # If no prediction, just write the record as is or skip?
                    # We'll write it through without refinement.
                    outfile.write(json.dumps(record) + "\n")
                    continue

                # Prepare the prompt for refinement
                # messages = [
                #     {"role": "user", "content": REFINE_PROMPT.format(input=prediction)},
                # ]

                try:
                    # response = client.chat.completions.create(model=MODEL, messages=messages)
                    response = client.models.generate_content(model=MODEL, contents=REFINE_PROMPT.format(input=prediction))

                    refined_content = response.text
                    record["refined_prediction"] = refined_content

                except Exception as e:
                    print(f"API call failed: {e}")
                    # We keep the record even if API fails, maybe mark it
                    record["error"] = str(e)

                outfile.write(json.dumps(record) + "\n")
                outfile.flush()

            except json.JSONDecodeError:
                print(f"Failed to decode JSON: {line[:50]}...")
            except Exception as e:
                print(f"Unexpected error: {e}")


if __name__ == "__main__":

    refine_jsonl()
