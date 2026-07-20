"""
utils.py

Utility functions for the RAG application.
Responsible for:
1. Reading text files
2. Chunking documents
"""

import os


def read_documents(folder_path: str):
    """
    Reads all .txt files from the given folder.

    Returns:
        List of dictionaries
        [
            {
                "filename": "...",
                "content": "..."
            }
        ]
    """

    documents = []

    for file in os.listdir(folder_path):

        if file.endswith(".txt"):

            path = os.path.join(folder_path, file)

            with open(path, "r", encoding="utf-8") as f:

                text = f.read()

                documents.append(
                    {
                        "filename": file,
                        "content": text
                    }
                )

    return documents


def chunk_text(text: str, chunk_size=500, overlap=100):
    """
    Splits text into overlapping chunks.

    Example

    Chunk 1 : 0 - 500

    Chunk 2 : 400 - 900

    Chunk 3 : 800 - 1300
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += (chunk_size - overlap)

    return chunks


def prepare_chunks(documents):
    """
    Converts every document into chunks.

    Returns

    [
        {
            "source":"CompanyPolicy.txt",
            "chunk":"Office Timing..."
        }
    ]
    """

    all_chunks = []

    for document in documents:

        chunks = chunk_text(document["content"])

        for chunk in chunks:

            all_chunks.append(
                {
                    "source": document["filename"],
                    "content": chunk
                }
            )

    return all_chunks


if __name__ == "__main__":

    docs = read_documents("data")

    print(f"Documents Loaded : {len(docs)}")

    chunks = prepare_chunks(docs)

    print(f"Chunks Created : {len(chunks)}")

    print()

    print(chunks[0])
