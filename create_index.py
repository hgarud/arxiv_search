"""Create index of Arxiv papers."""
import os
from dotenv import load_dotenv
load_dotenv()
import json
import logging

from openai import OpenAI
import pinecone


def get_pinecone_index(name: str, dimension: int, metric: str, pod_type: str):
    """Get a Pinecone index if it exists, and create it if not."""
    indexes = pinecone.list_indexes()
    if name not in indexes:
        pinecone.create_index(name, dimension=dimension, metric=metric, pod_type=pod_type)
    return pinecone.Index(name)

    return

def main():
    # Setup
    openai_client = OpenAI()
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east1-gcp")
    index = get_pinecone_index('papers', 1536, 'cosine', 'p1')

    # Create embeddings for all papers in the dataset
    query_cats = set(['cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.NE', 'cs.RO'])
    embeddings = {}
    logging.info('Collecting papers...')
    with open('arxiv-metadata-oai-snapshot.json', 'r') as f:
        for i, paper in enumerate(f):
            data = json.loads(paper)
            categories = set(data['categories'].split(' '))
            if not categories.isdisjoint(query_cats):
                pid = data['id']
                title = data['title'].strip().replace('\n', ' ')
                abstract = data['abstract'].strip().replace('\n', ' ')
                text = title + ' ' + abstract
                response = openai_client.embeddings.create(input = text, model='text-embedding-ada-002')
                embeddings[pid] = response.data[0].embedding
    
    # Add embeddings to Pinecone index
    for pid, embedding in embeddings.items():
        index.upsert(str(pid), embedding)




if __name__ == '__main__':
    main()