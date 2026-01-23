import requests
import os
import base64
import dotenv

dotenv.load_dotenv()

url = "https://api.fireworks.ai/inference/v1/chat/completions"


token = os.getenv("FIREWORKS_API_KEY")
model = "accounts/fireworks/models/qwen3-vl-235b-a22b-thinking"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


SYSTEM_PROMPT = """
You are a deep-thinking visual reasoner.  
All reasoning inside <think> ... </think> MUST strictly follow the Thinking Protocol below.  
Never skip steps. Never look at the image before planning.

# Thinking Protocol

1. PLAN BEFORE LOOKING  
 - When <think> begins, you must NOT inspect the image yet.  
 - First restate the task in 1-2 sentences.  
 - Then decompose the task into a short sequence of *atomic, executable steps* (2-6 steps).  
 - Each step must describe a specific operation, such as:  
      - "Identify all animals in the scene."  
      - "Check whether the man is riding any of them."  
      - "Compare the price tags."  
      - "Count the bottles on the top three shelves."

2. STEP-WISE VISUAL REASONING  
 - After completing the plan, inspect the image *step by step*, never all at once.  
 - For each step in your plan, do the following:
      a. Announce the step (e.g., "Now I should identify candidate animals").  
      b. Look at the image ONLY for information required for that step.  
         Do NOT give a general scene description.  
      c. Extract precise visual evidence directly relevant to the step.  
      d. Perform a **local verification**, such as:
           - "Are there other possible candidates?"  
           - "Could this object be interpreted differently?"  
           - "Is the relation ambiguous?"  
      e. Resolve ambiguities before moving to the next step.

3. FINAL SYNTHESIS  
 - After all steps are executed and verified, derive the final answer concisely.  
 - Do NOT repeat generic descriptions.  
 - The final answer should reflect the validated reasoning.

# Format Requirements

Your <think> must contain:
1. A Planning section (before any visual inspection).  
2. A Step-by-step execution section with local verification for each step.  
3. A final synthesis leading to the answer.
Your <think> style must follow:
1. Use a first-person narrative tone throughout your reasoning.
2. Do not use explicit headers or labels for the Planning, Execution, and Synthesis sections; instead, weave these stages into a single, continuous narrative flow.

After the reasoning inside <think>, output the final answer concisely.
"""

# image_path = "data/image-1d100e9-3.jpg"
image_base64 = encode_image("data/image-1d100e9-3.jpg")
image_base64_2 = encode_image("data/image-1d300ea.jpg")

USER_PROMPT = """
Image 1 shows the three sample types of screws for reference. Based on this reference, we assign class to the screws as follows: largest=A, middle=B, small=C. Now take a look at image 2.

In image 2, count the number of class A screws.
""".strip()

messages = [
    # {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64_2}"},
            },
            {"type": "text", "text": USER_PROMPT},
        ],
    },
]

payload = {
    "model": model,
    "messages": messages,
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 40,
    "stream": False,
}
headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

print(type(response.text))

import json

item = json.loads(response.text)
print(item["choices"][0]["message"]["content"])
