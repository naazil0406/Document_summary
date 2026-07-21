from qdrant_client import QdrantClient

client = QdrantClient("http://localhost:6333")

client.delete_collection("company_docs")

print("Deleted")