
from google import genai
import re
import json
def response_gen(query):
    client = genai.Client(api_key="AIzaSyDQPm0ZgpBHm9fRCGESZwpZ2TTeVTl1qxY")

    
    prompt = f"""
    Extract and return a raw JavaScript object with:
    - "context": short summary
    - "sentiment": "..."
    - "tags": min 5 keywords + 10 background knowledge terms and more of required 

    Only return raw object. No markdown or extra text.

    Input: "{query}"
    """



    response = client.models.generate_content(
        model="gemini-2.0-flash",
            contents=prompt,
            )
    
    match = re.search(r'\{.*\}', response.text, re.DOTALL)
    if match:
        clean = match.group(0)
        try:
            metadata = json.loads(clean)
            if all(k in metadata for k in ("context", "sentiment", "tags")):
                return metadata
        except Exception as e:
            print("[!] Regex parse also failed")



print(response_gen("mujhe chess bahoot psnd h me ek district level ka chess player bhi hu"))