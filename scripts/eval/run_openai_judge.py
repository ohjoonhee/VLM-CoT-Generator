import json
import os
from openai import OpenAI
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()

# Files
INPUT_FILE = "output/refined_reasoning_v3.jsonl"
OUTPUT_FILE = "output/judge_filtered_reasoning_v3_v2.jsonl"

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


def refine_jsonl():
    client = OpenAI(api_key=API_KEY)

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
                prediction = record.get("refined_prediction", "")

                if not prediction:
                    # If no prediction, just write the record as is or skip?
                    # We'll write it through without refinement.
                    outfile.write(json.dumps(record) + "\n")
                    continue

                # Prepare the prompt for refinement
                messages = [
                    {"role": "user", "content": BINARY_JUDGE_PROMPT.format(question=record["question"], answer=record["answer"], prediction=prediction, positive="1", negative="0")},
                ]

                try:
                    response = client.chat.completions.create(model=MODEL, messages=messages)

                    judge_result = response.choices[0].message.content
                    record["judge_result"] = judge_result

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
