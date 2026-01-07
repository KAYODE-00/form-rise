from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()

app = FastAPI()

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


class Document(BaseModel):
    text: str


@app.post("/extract")
def extract_data(doc: Document):
    prompt = f"""Extract the following fields from this document, and return ONLY valid JSON, nothing else:
- invoice_number
- date
- bill_to
- amount
- due_date

If a field is missing, use null.

Document:
{doc.text}
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    raw_output = response.choices[0].message.content

    try:
        extracted = json.loads(raw_output)
    except json.JSONDecodeError:
        extracted = {"error": "Model did not return valid JSON", "raw": raw_output}

    return extracted