from openai import Client

BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"


def main():
    client = Client(base_url=BASE_URL, api_key=API_KEY)

    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-32B-Thinking-FP8",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        max_tokens=1024,
        temperature=0.7,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
