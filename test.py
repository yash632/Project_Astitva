# pip install transformers[torch] sentence-transformers torch redis pinecone numpy accelerator

# huggingface-cli login
# docker run -p 6379:6379 redis
# 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
# runner commands

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import torch
import redis
from pinecone import Pinecone
import uuid
import json
from numpy import dot
from numpy.linalg import norm

# ==== Setup Section ====

# GPT-Neo 1.3B Semantic Pipeline
print("[+] Loading GPT-Neo 1.3B model...")
device = 0 if torch.cuda.is_available() else -1

# Loading GPT-Neo 1.3B
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

gpt_neo = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Embedding Model
print("[+] Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Redis (Short-Term Memory)
print("[+] Connecting to Redis...")
rdb = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Pinecone (Long-Term Memory)
print("[+] Connecting to Pinecone...")
pc = Pinecone(api_key="your_pinecone_api_key", environment="gcp-starter")
index_name = "project-astitva-ltm"
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine"
    )

index = pc.Index(index_name)

# ==== Core Functions ====

def extract_semantic_info(text):
    """
    Extract entities, sentiment, and context from the input text using GPT-Neo 1.3B.
    """
    prompt = f"""Q: Extract key entities, their types, context, and sentiment from:\n\"{text}\"\nA:"""
    output = gpt_neo(prompt, max_new_tokens=100, do_sample=False)[0]['generated_text']
    
    # Extract the context from the output, assuming the model gives a structured output
    semantic_info = output.split("A:")[-1].strip()
    
    # We can further structure this output into context, sentiment, and other details if needed.
    # Here, I'll return the full response as context for now.
    return semantic_info

def get_embedding(text):
    return embedder.encode(text).tolist()

def store_to_redis(key, text, embedding, semantic):
    rdb.hset(key, mapping={
        "text": text,
        "embedding": json.dumps(embedding),
        "semantic": semantic
    })

def store_to_pinecone(id, vector, metadata):
    index.upsert([(id, vector, metadata)])

def save_memory(text):
    """
    Save the memory (text, its embedding, and semantic data) into Redis and Pinecone.
    """
    semantic = extract_semantic_info(text)  # Get semantic info (which includes context)
    vector = get_embedding(text)  # Get the vector embedding for the text
    mem_id = str(uuid.uuid4())  # Unique ID for the memory
    
    # Store to Redis and Pinecone
    store_to_redis(mem_id, text, vector, semantic)
    store_to_pinecone(mem_id, vector, {"semantic": semantic, "text": text})
    print(f"[+] Memory saved: {mem_id}")

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def retrieve_memory(query, top_k=5):
    """
    Retrieve memory based on the input query.
    """
    q_vector = get_embedding(query)  # Get the vector for the query
    results = index.query(vector=q_vector, top_k=top_k, include_metadata=True)  # Query Pinecone
    return results

def should_clarify(query, score, threshold=0.55):
    """
    Determine if a clarification question should be asked.
    """
    return score < threshold

# ==== Demo Usage ====

if __name__ == "__main__":
    print("\n[+] Saving a memory example...")
    save_memory("Mujhe Python aur JavaScript pasand hain")
    save_memory("Mujhe React aur Node.js aata hai")
    save_memory("Mujhe Photoshop aur Figma bhi aata hai")

    print("\n[+] Querying a memory...")
    query = "Tujhe konsi skills aati hain?"
    results = retrieve_memory(query, top_k=5)

    combined_skills = []
    for match in results['matches']:
        semantic = match['metadata']['semantic']
        if "Programming Language" in semantic or "TechSkill" in semantic:
            combined_skills.append(match['metadata']['text'])

    print("\n-- Combined Relevant Skills --")
    if combined_skills:
        for skill in combined_skills:
            print("-", skill)
    else:
        print("No strong matches found. Consider clarification.")

    # Optional: use max score for should_clarify
    max_score = max([match['score'] for match in results['matches']])
    print("\nClarify Needed:", should_clarify(query, max_score))
