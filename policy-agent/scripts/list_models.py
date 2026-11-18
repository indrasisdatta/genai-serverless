#!/usr/bin/env python3
"""Script to verify Groq models and structured output."""

import os
import sys
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq


load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

print("Available models:")
for m in client.models.list().data:
    print(f"  {m.id}")

# Test structured output
print("\nTesting structured output...")


class Verdict(BaseModel):
    relevant: bool
    reason: str


llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

try:
    result = llm.with_structured_output(Verdict).invoke(
        "Is 'roaming charges apply outside the US' relevant to 'Do I pay extra in Canada?'"
    )
    print(f"Structured output works: {result}")
except Exception as e:
    print(f"Standard structured output failed: {e}")
    print("Trying json_mode...")
    try:
        result = llm.with_structured_output(Verdict, method="json_mode").invoke(
            "Is 'roaming charges apply outside the US' relevant to 'Do I pay extra in Canada?'"
        )
        print(f"json_mode works: {result}")
    except Exception as e2:
        print(f"json_mode also failed: {e2}")
        sys.exit(1)