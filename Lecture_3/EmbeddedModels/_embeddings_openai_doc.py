from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "Delhi is the capital of India",
    "Jaipur is capital of Rajasthan",
    "Tomorrow is Sunday"
]

result = embedding.embed_documents(documents)
print(result)