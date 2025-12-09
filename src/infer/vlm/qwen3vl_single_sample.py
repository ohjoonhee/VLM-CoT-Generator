import base64
from openai import Client

BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"

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

# 1. Always start with a high-level plan of action.


def load_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    client = Client(base_url=BASE_URL, api_key=API_KEY)
    base64_image = load_image("data/700.jpg")

    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-8B-Thinking",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": "What is the sum of the sales for Item 1 and Item 6?\nA. 97\nB. 96\nC. 95\nD. 94\n"},
                ],
            },
        ],
        max_tokens=4096,
        temperature=0.7,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
