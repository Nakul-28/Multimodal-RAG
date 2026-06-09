from chromadb import PersistentClient

client = PersistentClient(path="chroma_db")

collections = client.list_collections()

print("Collections Found:", len(collections))

for collection in collections:
    print("\nCollection Name:", collection.name)
    print("Document Count:", collection.count())