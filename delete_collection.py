from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.delete_collection(collection_name="{collection_name}")

print("success")