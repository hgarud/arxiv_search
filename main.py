"""Main API script for the AI Research Paper Search Engine."""

import argparse
import os
from typing import Union
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import pinecone

from fastapi import FastAPI
app = FastAPI()


@app.get("/search/{query}")
def main(query: str):
    # Setup
    openai_client = OpenAI()
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east1-gcp")
    index = pinecone.Index("papers")

    # Create embeddings for query
    response = openai_client.embeddings.create(input = query, model='text-embedding-ada-002')
    query_embedding = response.data[0].embedding

    # Search for similar papers
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_values=False,
        namespace='main'
    )
    return [paper['id'] for paper in results['matches']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI Research Paper Search Engine')
    parser.add_argument('--query', type=str, help='Query string')

    args = parser.parse_args()

    print(main(args.query))
