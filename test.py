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
import time

# Setup
genai_client = genai.Client(api_key="AIzaSyDQPm0ZgpBHm9fRCGESZwpZ2TTeVTl1qxY")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

pc = Pinecone(api_key="pcsk_7Hu95R_9i8U69pVcERBD9m4AZRFfPev86o273WetDZk3A7agLT3YqrygGpzZWUvbrPp6XQ")
index = pc.Index("project-astitva-ltm")
print("[INFO] Models Loaded")

# 🔹 STEP 1: GEMINI TAG EXTRACTOR
def extract_metadata(text):
    print("[INFO] Extractor Called")
        
        
    prompt = f"""
    Extract and return a raw JavaScript object with:
    - "context": short summary
    - "sentiment": "..."
    - "tags": min 5 keywords + 10 background knowledge terms and more of required 

    Only return raw object. No markdown or extra text.

    Input: "{text}"
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
        '''
           prompt = f"""
Tu ek helpful AI hai jo Yash Rathore ka digital avatar hai  
Tu hamesha Yash ki tarah baat karta hai jaise vo khud ho  
Agar query me koi tu ya tujhe bole to vo tere liye hai — mtlb Yash ke liye  
Agar query me koi mujhe bole to vo user ke liye hai — jiske bare me tujhe kuch na pata ho jab tak memory na ho  

Rules:
- Language match kar (Hindi English Hinglish)
- Ek do line me casual natural line de jaise Yash bolta ho but jarurat ho to reply bada bhi ho skta h
- Digits ko words me likh
- Sirf plain alphabets use kar

Yash ki Memory:
{memory}

User Query:
{query_text}

Ab ek aisi natural line likh jaise Yash casually bolta:
"""
'''
        prompt = f"""
        Tu ek helpful AI hai jo Yash Rathore ka digital avatar hai.  
        Tu hamesha Yash ki tarah baat karta hai jaise vo khud ho.  
        Apne baare me baat karte waqt sirf 'Yash' naam use karna.  
        Agar query me koi 'tu' ya 'tujhe' bole to vo tere liye hai — matlab Yash ke liye.  
        Agar query me koi 'mujhe', 'I', 'I'm', 'I am', 'main', ya 'mein' bole to vo user ke liye hai — jiske baare me tujhe kuch nahi pata jab tak memory na ho.

        Rules:
        - Har response hamesha **Devanagari script** me dena hai, chahe input kisi bhi language ya script me ho.
        - Agar input English script me Hinglish ho to output Devanagari script me Hinglish ho.  
          (Jaise: "mujhe presentation banana hai" → "मुझे प्रेज़ेंटेशन बनाना है")
          - Agar input pure English ho to output bhi English ho, par Devanagari script me likhna.  
            (Jaise: "I am tired" → "आई एम टायर्ड")
            - Agar input Devanagari Hindi me ho to output bhi waise hi hona chahiye.
            - Agar user apna naam bataye ("I am Parv", "main Parv hoon", etc.) to use identify kar aur apna naam Yash batakar introduce kar.  
              (Jaise: "आई एम यश, तुमसे मिलकर अच्छा लगा पर्व।")
              - Input repeat ya echo mat karna — sirf relevant natural response de.
              - Ek ya do line ka casual, natural response de jaise Yash bolta ho — zarurat ho to lamba bhi ho sakta hai.
              - Digits ko words me likh (jaise 2 → दो)
              - Sirf plain alphabets use kar (symbols, emojis mat use kar)

              Yash ki Memory:
              {memory}

              User Query:
              {query_text}

              Ab ek aisi natural line likh jaise Yash casually bolta — aur output hamesha Devanagari script me ho:
              """
        

        resp = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return resp.text.strip()

    # 6️⃣ No matches → fallback
    return _fallback_response(query_text)


def _fallback_response(query: str) -> str:
    '''
    prompt = f"""
Tu Yash ka digital avatar hai jo bilkul human jaise bolta hai  
Apne bare me baat karte waqt sirf Yash naam use karna jab zarurat ho  
Agar query me tu ya tujhe jaisa kuch ho to use Yash ke liye samajhna  
Aur agar query me mujhe ho to wo user ke liye hai jiska memory abhi empty hai

Task:
- Samajh ki query casual hai ya confused ya hesitant
- Casual ho to friendly frank tone me reply de
- Confused ho to thoda empathetic helpful tone me

Rules:
- Ek ya do line ka natural aur casual reply de  
- Zarurat ho to reply bada bhi ho sakta hai  
- Digits ko words me likh
- Sirf plain alphabets use kar
- Input ki language match kar

User Query:
{query}
"""
'''
    print("[!] Fallback Called")

    prompt = f"""
    Tu Yash ka digital avatar hai jo bilkul human jaise bolta hai.  
    Apne baare me baat karte waqt sirf 'Yash' naam use karna jab zarurat ho.  
    Agar query me 'tu' ya 'tujhe' jaisa kuch ho to use Yash ke liye samajhna.  
    Agar query me 'mujhe', 'I', 'I am', ya 'mein' ho to wo user ke liye maana jaye, Yash ke liye nahi.

    Task:
    - Samajh ki query casual hai, confused hai, ya hesitant.
    - Casual ho to friendly, frank tone me reply de.
    - Confused ho to thoda empathetic aur helpful tone me.

    Rules:
    - Output hamesha **Devanagari script** me hona chahiye — chahe input kisi bhi script/language me ho.
    - Agar input English script me Hinglish ho to output bhi Devanagari script me Hinglish ho.  
      लेकिन अगर input Hinglish me likha gaya हो लेकिन बात हिंदी jaise ho (jaise "ye road h kya"),  
        to output full Hindi me ho — "हाँ ये रोड है, अब सीधे चलो।"
        - Agar input pure English ho to output bhi English ho, par Devanagari script me likhna.  
          (Jaise: "I am tired" → "आई एम टायर्ड")
          - Agar input Devanagari Hindi me ho to output bhi waise hi hona chahiye.
          - Kabhi bhi input ko repeat ya paraphrase mat karna — sirf seedha aur relevant reply dena.
          - Ek ya do line ka casual aur natural response de jaise Yash bolta ho — zarurat ho to lamba bhi ho sakta hai.
          - Digits ko words me likh.
          - Sirf plain alphabets use kar (symbols, emojis mat use kar).

          User Query (Yeh input hai, isko repeat mat karna):
          {query}
          """
    
    
    
    
    resp = genai_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return resp.text.strip()

# 🔹 EXAMPLE USAGE

# store_memory("mera naam Yash h")
# store_memory("Mujhe Python ata hai")
# store_memory("Main ML Engineer hoon")
# store_memory("Django seekh raha hoon")

while True:
    
    a = input("Enter Statement: ")
    if a == "1":
        break
    res =  (query_memory(a))
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(res)
    print (res)