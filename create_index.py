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
    summarize_prompt = "Please summarize the given abstract of an AI research paper so that a search engine can retrieve relevant papers based on cosine similarity between vector embeddings of the summary and input user query. Be extremely frugal with output words. Do not include any results. Abstract:"

    # Create embeddings for all papers in the dataset
    query_cats = set(['cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.NE', 'cs.RO'])
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
                print("Processing paper id {pid}")

                # Create title+abstract embeddings
                response = openai_client.embeddings.create(input = text, model='text-embedding-ada-002')
                main_embedding = response.data[0].embedding

                # Create summary of abstract
                response = openai_client.chat.completions.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": summarize_prompt},
                                    {"role": "user", "content": abstract}
                                ]
                            )
                summary =  response.choices[0].message.content
                # Create summary embeddings
                response = openai_client.embeddings.create(input = summary, model='text-embedding-ada-002')
                summary_embedding = response.data[0].embedding

                # Upsert embeddings to Pinecone index
                index.upsert(
                    vectors=[{"id": str(pid), "values": main_embedding}],
                    namespace='main'
                )
                index.upsert(
                    vectors=[
                        {
                            "id": str(pid),
                            "values": summary_embedding,
                            "metadata": {"summary": summary}
                        }],
                    namespace='summary'
                )


if __name__ == '__main__':
    main()