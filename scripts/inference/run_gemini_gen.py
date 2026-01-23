import json
import os
from google import genai
from google.genai import types
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()


# Model configuration
MODEL = "gemini-3-flash-preview"

client = genai.Client()
prompt = "What is the sum of the first 50 prime numbers?"
response = client.models.generate_content(
    model=MODEL,
    contents=prompt,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="medium"),
    ),
)

for part in response.candidates[0].content.parts:
    if not part.text:
        continue
    if part.thought:
        print("Thought summary:")
        print(part.text)
        print()
    else:
        print("Answer:")
        print(part.text)
        print()
