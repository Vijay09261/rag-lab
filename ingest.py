
"""
ingest.py

Reads documents from the data folder,
creates chunks,
generates embeddings using Azure OpenAI.
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

from utils import read_documents, prepare_chunks

# Load environment variables
load_dotenv()

# Azure OpenAI Client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_DEPLOYMENT")


def generate_embedding(text):
    """
    Generate an embedding for the given text.
    """

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


def main():

    print("=" * 50)
    print("Loading Documents...")
    print("=" * 50)

    documents = read_documents("data")

    print(f"Documents Loaded : {len(documents)}")

    chunks = prepare_chunks(documents)

    print(f"Chunks Created : {len(chunks)}")

    print("\nGenerating Embeddings...\n")

    embedded_chunks = []

    for i, chunk in enumerate(chunks):

        vector = generate_embedding(chunk["content"])

        embedded_chunks.append({
            "id": str(i),
            "source": chunk["source"],
            "content": chunk["content"],
            "embedding": vector
        })

        print(f"Processed Chunk {i + 1}/{len(chunks)}")

    print("\nCompleted Successfully!")

    print(f"Total Embedded Chunks : {len(embedded_chunks)}")

    # Display one sample
    print("\nSample Record:")
    print("----------------------")
    print("Source :", embedded_chunks[0]["source"])
    print("Content:", embedded_chunks[0]["content"][:100], "...")
    print("Embedding Length:", len(embedded_chunks[0]["embedding"]))


if __name__ == "__main__":
    main()
