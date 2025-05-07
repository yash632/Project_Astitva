
from google import genai

client = genai.Client(api_key="AIzaSyDQPm0ZgpBHm9fRCGESZwpZ2TTeVTl1qxY")

query = "mujhe kya pasand h"
prompt = f"""
You are an AI that extracts structured semantic information. 

Given the input text below, return a JSON object with the following keys only: 
- "context": a short summary of the input
- "sentiment": one of ["positive", "neutral", "negative"]
- "tags": intent of the sentence

Important:
- DO NOT include any markdown, code block, or triple quotes.
- Output must be valid raw JSON only.

Input:
"{query}"

Output:
"""



response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
)

print(response.text)