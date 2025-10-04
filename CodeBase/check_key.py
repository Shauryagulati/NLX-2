from openai import OpenAI
import os


def check_openai_key():
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, can you confirm my API key works?"}],
            max_tokens=20
        )
        print("OpenAI key is working!")
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print("OpenAI key not working.")
        print("Error:", str(e))

if __name__ == "__main__":
    check_openai_key()