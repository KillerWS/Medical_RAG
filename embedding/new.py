from transformers import AutoTokenizer, AutoModel
import torch
from langchain_community.vectorstores import DocArrayInMemorySearch
from docarray import Document, DocumentArray

# Load your model and tokenizer here
model_name = "../models/BAAI/bge-large-zh-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Your dataset texts
texts = [
    "青蛙是食草动物",
    "人是由恐龙进化而来的。",
    "熊猫喜欢吃天鹅肉。",
    "1+1=5",
    "2+2=8",
    "3+3=9",
    "Gemini Pro is a Large Language Model was made by GoogleDeepMind",
    "A Language model is trained by predicting the next token"
]

# Generate embeddings for your texts
text_embeddings = [get_embedding(text) for text in texts]


# vectordb = DocArrayInMemorySearch.from_texts(texts)

# Assuming you have a way to create or update your vector database with these embeddings
# vectordb.update_with_embeddings(text_embeddings)



# results = vectordb.search(query_embedding, top_k=5)
# print(results)

# Create documents with embeddings
document_embeddings = [text_embeddings] # Your document embeddings here
documents = [Document(embedding=emb) for emb in document_embeddings]
doc_array = DocumentArray(documents)

# Embed your query and search
query_embedding = get_embedding("language model") # Your query embedding here
query_doc = Document(embedding=query_embedding)

# Perform the search
doc_array.match(query_doc, limit=3)

# The first three matches are now stored in query_doc.matches
for match in query_doc.matches[:3]:
    print(match.text)  # Assuming you've also stored the text in each Document
