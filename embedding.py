from openai import OpenAI
import pandas as pd
import os

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

csv_file_path = 'image_summaries.csv'
df = pd.read_csv(csv_file_path)

df['ada_embedding'] = df['Description'].apply(lambda x: get_embedding(x, model='text-embedding-3-small'))

output_csv_path = 'embedded_descriptions.csv'
df.to_csv(output_csv_path, index=False)

print(f"Embeddings saved to {output_csv_path}")