from docarray import Document, DocumentArray
from transformers import AutoTokenizer, AutoModel
import torch
# Mock function to represent generating an embedding for a given text.
# Replace this with your actual function to generate embeddings with bge-large-zh-v1.5
model_name = "../models/BAAI/bge-large-zh-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    # truncation表示截断策略，max_length=512 指定了编码输入中允许的最大标记数, padding=True 可以确保所有编码输入具有相同的长度，这对于批量处理非常重要。
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Your dataset
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

# Generate embeddings for each text in your dataset
documents = [Document(text=text, embedding=get_embedding(text)) for text in texts]

# Create a DocumentArray and add your documents
doc_array = DocumentArray(documents)


# Embed your query
query_text = "language model"
query_embedding = get_embedding(query_text)
query_doc = Document(embedding=query_embedding)

# Perform the search
doc_array.match(query_doc, limit=3)

# Print the top 3 matching texts
for match in query_doc.matches[:3]:
    print(match.text)
