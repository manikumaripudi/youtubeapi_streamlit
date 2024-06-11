from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="genai-docs",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
)

print("Success")