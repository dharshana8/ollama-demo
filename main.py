import ollama

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

VECTOR_DB = []

def load_dataset(filepath='cat-facts.txt'):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.readlines()

def build_vector_db(dataset):
    VECTOR_DB.clear()
    for chunk in dataset:
        embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=chunk)['embedding']
        VECTOR_DB.append((chunk, embedding))

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return 0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)

def retrieve(query, top_n=3):
    query_embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)['embedding']
    similarities = [(chunk, cosine_similarity(query_embedding, emb)) for chunk, emb in VECTOR_DB]
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

def stream_response(query):
    results = retrieve(query)
    context = "\n".join(f' - {chunk.strip()}' for chunk, _ in results)
    instruction_prompt = f"""You are a helpful chatbot.
Use only the following pieces of context to answer the question.
Don't make up any new information.

Context:
{context}
"""
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': query},
        ],
        stream=True,
    )
    for chunk in stream:
        yield chunk['message']['content']

def get_sources(query):
    return [(chunk.strip(), round(score, 3)) for chunk, score in retrieve(query)]
