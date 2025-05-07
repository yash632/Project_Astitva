# pip install transformers[torch] sentence-transformers torch redis pinecone numpy accelerator

# huggingface-cli login
# docker run -p 6379:6379 redis
# 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
# pcsk_7Hu95R_9i8U69pVcERBD9m4AZRFfPev86o273WetDZk3A7agLT3YqrygGpzZWUvbrPp6XQ
# runner commands
# https://didactic-barnacle-v665v7jp45g4h6prr-6379.app.github.dev/





from google import genai
from sentence_transformers import SentenceTransformer
import redis
from pinecone import Pinecone
import json
import re


# Setup
genai_client = genai.Client(api_key="AIzaSyDQPm0ZgpBHm9fRCGESZwpZ2TTeVTl1qxY")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

pc = Pinecone(api_key="pcsk_7Hu95R_9i8U69pVcERBD9m4AZRFfPev86o273WetDZk3A7agLT3YqrygGpzZWUvbrPp6XQ")
index = pc.Index("project-astitva-ltm")
print("[INFO] Models Loaded")

# 🔹 STEP 1: GEMINI TAG EXTRACTOR
def extract_metadata(text):
    print("[INFO] Extractor Called")
    prompt = f"""
You are an AI that extracts structured semantic information. 

Given the input text below, return a Java script object with the following keys only: 
- "context": a short summary of the input
- "sentiment": one of ["positive", "neutral", "negative"]
- "tags": a list of only relevant keywords (minimum 5) and background knowledge keywords (minimum 10)

Important:
- DO NOT include any markdown, code block, or triple quotes.
- Output must be valid raw Java script object only.

Input:
"{text}"

Output:
"""
    response = genai_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    raw = response.text.strip()
    

    # ✅ STEP 1: Try direct JSON parsing
    try:
        metadata = json.loads(raw)
        if all(k in metadata for k in ("context", "sentiment", "tags")):
            return metadata
    except Exception as e:
        print("[!] Direct parse failed")

    # 🧩 STEP 2: Fallback - try regex to extract the JSON part only
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        clean = match.group(0)
        try:
            metadata = json.loads(clean)
            if all(k in metadata for k in ("context", "sentiment", "tags")):
                return metadata
        except Exception as e:
            print("[!] Regex parse also failed")

    # ❌ STEP 3: Return default on failure
    print("[!] Regex Fallback Also Failed Returning Default Meta Data")
    return {"context": "", "sentiment": "neutral", "tags": []}

# 🔹 STEP 2: STORE MEMORY
'''
def store_memory(text):
    print("[INFO] Store Memory Called")
    meta = extract_metadata(text)
    vector = embedder.encode(text).tolist()
    memory_id = f"memory:{hash(text)}"

    # Redis store
    redis_client.hset(memory_id, mapping={
        "text": text,
        "tags": json.dumps(meta["tags"]),
        "context": meta["context"],
        "embedding": json.dumps(vector)
    })

    # Pinecone store
    index.upsert([(memory_id, vector, {"text": text, "tags": meta["tags"]})])
'''

def store_memory(text):
    print("[INFO] Store Memory Called")
    vector = embedder.encode(text).tolist()

    # 🔍 Check for similar memory in Pinecone
    result = index.query(vector=vector, top_k=3, include_metadata=True)
    for match in result["matches"]:
        similarity = match["score"]  # 1.0 = perfect match
        if similarity > 0.92:  # adjust threshold as needed
            print(f"[INFO] Similar memory already exists (Score: {similarity:.2f}). Skipping store.")
            return
    meta = extract_metadata(text)
    memory_id = f"memory:{hash(text)}"


    # ✅ Pinecone store (long-term)
    index.upsert([(memory_id, vector, {"text": text, "tags": meta["tags"]})])

# 🔹 STEP 3: QUERY MEMORY
def query_memory(query_text):
    print("[INFO] Querying Called")

    # 1️⃣ Extract semantics from the query
    meta = extract_metadata(query_text)
    query_context = meta.get("context", query_text)
    query_tags    = meta.get("tags", [])

    # 2️⃣ If no tags, skip directly to fallback
    if not query_tags:
        print("[INFO] No tags extracted — skipping memory search.")
        return _fallback_response(query_text)

    # 3️⃣ Build Pinecone filter to only return memories with ANY of those tags
    pinecone_filter = {"tags": {"$in": query_tags}}

    # 4️⃣ Query Pinecone with both metadata filter and vector ranking
    query_vec = embedder.encode(query_context).tolist()
    result = index.query(
        vector=query_vec,
        top_k=10,
        include_metadata=True,
        filter=pinecone_filter
    )

    # 5️⃣ Collect the stored sentences
    matches = [m["metadata"]["text"] for m in result["matches"]]


    if matches:
        memory = ",".join(f" {text}" for text in matches)

        print("results found: ",memory,"\n")
        prompt = f"""
        You are a helpful AI assistant that speaks like a human. You are the digital avatar of YASH RATHORE.

        Important: Never mention your identity unless absolutely necessary. If needed, use only the name Yash — never the full name or any title.

        Below is the user's memory, and the query_text they just said. Your job is to:
        - Understand the intent, context, and emotional behavior based on the query_text.
        - Then generate one fluent, short, natural sentence that sounds like something Yash would casually say in a real conversation.

        Rules:
        - Match the language of the input query (Hindi English Hinglish)
        - Keep the reply short and natural (like a friend)
        - Do NOT use any punctuation (no dots, commas, emojis)
        - Do NOT use digits — write number words like one two three
        - Use only plain alphabets

        User Memory (facts and tone preferences):
        {memory}

        User Query:
        {query_text}

        Now write a natural sentence that reflects how Yash would casually respond:
        """


        resp = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return resp.text.strip()

    # 6️⃣ No matches → fallback
    return _fallback_response(query_text)


def _fallback_response(query: str) -> str:
    prompt = f"""
    You are a human-like AI assistant and the digital avatar of YASH RATHORE. Do NOT mention your name or identity in the response unless absolutely necessary. If needed, only use the name Yash — never use full name or titles.

    The user has asked a question, but there is no memory available to answer from.

    Your task:
    - Analyze the query and determine whether it sounds casual, serious, confused, or hesitant.
    - If it sounds casual or friendly, respond with a short frank sentence just like a real person would talk to a friend.
    - If the query shows confusion or hesitation, reply in a slightly empathetic way like someone trying to help without being too formal.

    Instructions:
    - Keep your reply short and natural (one or two lines max)
    - Do NOT use any punctuation marks like dot comma or emojis
    - Do NOT use any digits use their word form like one two three
    - Use only plain alphabets
    - Match the language of the user input (Hindi English Hinglish)

    User Query:
    "{query}"
    """


    resp = genai_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return resp.text.strip()

# 🔹 EXAMPLE USAGE

# store_memory("Mujhe Python achha lagta hai")
# store_memory("Mujhe Python ata hai")
# store_memory("Main ML Engineer hoon")
# store_memory("Django seekh raha hoon")


print(query_memory("yrr kuchh developement ke baare me bata na"))
