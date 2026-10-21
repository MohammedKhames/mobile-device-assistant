import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from app.prompts.templates import SYSTEM_PROMPT, build_user_prompt
from app.schema.device import Device
from app.utils.security import check_prompt_injection


# Load the Groq key and model name from the project .env file.
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")


def extract_device(text: str) -> Device:
    # Reject suspicious instructions before making an LLM request.
    check_prompt_injection(text)
    
    # Keep the original instructions and device text in the conversation.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text)},
    ]

    # One initial attempt and up to two retries.
    for attempt in range(3):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
        )

        raw_output = response.choices[0].message.content or ""

        try:
            # Parse the JSON text, then validate its fields and values.
            data = json.loads(raw_output)
            return Device.model_validate(data)

        except (json.JSONDecodeError, ValidationError) as error:
            # Stop after the final failed attempt.
            if attempt == 2:
                raise

            # Show the model its previous answer and the validation error.
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user","content": (
                    f"Your previous response was invalid: {error}. "
                    "Return a corrected JSON object matching the schema only."
                ),
            })
