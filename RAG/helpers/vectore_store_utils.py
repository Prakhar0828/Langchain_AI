import os
import hashlib
from pinecone import Pinecone, ServerlessSpec

def generate_document_id(text: str) -> str:
    """Generate a deterministic ID by hashing the document text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def add_documents_to_pinecone_index(vector_store, documents, batch_size=10):
    """Add documents to the Pinecone index in batches with deterministic IDs."""
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        ids = [generate_document_id(text) for text in texts]
        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"Successfully added documents {i} to {i + len(batch) - 1} to Pinecone index.")

def create_pinecone_index(index_name):
    """Create a Pinecone index if it doesn't exist."""
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    index_list = [d['name'] for d in pc.list_indexes()]
    if index_name not in index_list:
        pc.create_index(name=index_name, dimension=768, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))  # Adjust dimension as needed
    return pc.Index(index_name)
