import os
import shutil
import json
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()
os.makedirs("data", exist_ok=True)

app = FastAPI()

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"


class Document(BaseModel):
    text: str


def extract_fields(text: str) -> dict:
    prompt = f"""Extract the following fields from this document, and return ONLY valid JSON, nothing else:
- invoice_number
- date
- bill_to
- amount
- due_date

If a field is missing, use null.

Document:
{text}
"""
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    raw_output = response.choices[0].message.content

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {"error": "Model did not return valid JSON", "raw": raw_output}


def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    print(f"Extracted text length: {len(text)}")
    print(repr(text[:300]))
    return text


@app.post("/extract")
def extract_data(doc: Document):
    return extract_fields(doc.text)


@app.post("/extract-pdf")
async def extract_from_pdf(file: UploadFile = File(...)):
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = read_pdf(file_path)
    return extract_fields(text)