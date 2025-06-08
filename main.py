from typing import List, Optional
from google import genai
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI
load_dotenv()
import json
import os


class SubRisk(BaseModel):
    riskClause: Optional[str] = None
    riskBrief: Optional[str] = None
    riskExplain: Optional[str] = None
    resultType: Optional[str] = None
    originalContent: Optional[str] = None
    resultContent: Optional[str] = None

class Result(BaseModel):
    examineResult: Optional[str] = None
    ruleTag: Optional[str] = None
    ruleTitle: Optional[str] = None
    examineBrief: Optional[str] = None
    riskLevel: Optional[str] = None
    subRisks: Optional[List[SubRisk]] = None
    ruleSequence: Optional[str] = None

class Output(BaseModel):
    result: Optional[Result] = None

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/enhance")
async def get_enhancement(document_details: Output):
    prompt = f"""You are an expert legal analyst.  
Given the following legal document, perform the following tasks:
1. Analyze the document and rewrite it to be more descriptive, detailed, and accurate.
2. Identify and explain any potential risks or important clauses within the document.
3. Rewrite the analysis and risk explanations in clear, simple language that can be easily understood by people without a legal background.

Please structure your response as follows:
- Improved and Expanded Document
- Key Risks and Clauses (with explanations in plain language)
- Summary for Non-Legal Readers
- Keep the original language and structure of the document intact as much as possible.

Here is the legal document:\n\n{document_details}"""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": Output,
        },
    )

    return json.loads(response.candidates[0].content.parts[0].text)
