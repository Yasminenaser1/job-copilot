"""Index profile/ (resume, projects) into ChromaDB."""
from pathlib import Path
import chromadb
import ollama

EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 400

def chunk_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) > CHUNK_SIZE and current:
            chunks.append(current)
            current = ""
        current += p + "\n\n"
    if current:
        chunks.append(current)
    return chunks

def embed(text: str) -> list[float]:
    return ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]

def main():
    client = chromadb.PersistentClient(path="db")
    try:
        client.delete_collection("profile")  # rebuild fresh each time
    except Exception:
        pass
    collection = client.create_collection("profile")

    for file in Path("profile").glob("*.md"):
        chunks = chunk_text(file.read_text())
        for i, chunk in enumerate(chunks):
            collection.add(
                ids=[f"{file.name}-{i}"],
                documents=[chunk],
                embeddings=[embed(chunk)],
                metadatas=[{"source": file.name}],
            )
        print(f"📄 {file.name}: {len(chunks)} chunk(s) indexed")
    print(f"\n✅ Profile indexed: {collection.count()} chunks")

if __name__ == "__main__":
    main()
