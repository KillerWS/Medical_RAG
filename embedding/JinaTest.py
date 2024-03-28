# 文本向量化过程

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from docarray import DocumentArray, Document
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 假设 "../models/BAAI/bge-large-zh-v1.5" 是您的模型路径
model_name = "../models/BAAI/bge-large-zh-v1.5"
# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(texts,tokenizer, model):
    """
    为给定的文本数组获取嵌入向量。
    
    参数:
    - texts (list of str): 要转换为嵌入向量的文本列表。
    - model_name (str): 预训练模型的路径或名称。
    
    返回:
    - torch.Tensor: 文本的嵌入向量，形状为 (len(texts), embedding_dimension)。
    """
    
    
    # 分词处理
    # 如果texts是一个字符串列表，分词器（Tokenizer）会自动处理这个列表中的所有文本。然后，它会返回一个批处理的输入，其中包含了所有文本的编码表示。这意味着所有文本都被一次性转换成了模型需要的格式，而不是逐个转换
    encoded_input = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
    
    # 生成嵌入向量
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # 获取所有文本的嵌入表示，这里使用最后一层的隐藏状态的平均值作为文本的嵌入向量
    embeddings = model_output.last_hidden_state.mean(dim=1)
    
    return embeddings

# 示例文本数组
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

embeddings = get_embedding(texts,tokenizer, model)
print(type(embeddings) )
for embedding in embeddings:
    print( + embedding)

print(embeddings.size())
query = "Langage model Google"

 
# 要使用 FAISS 进行搜索，您需要将 torch.Tensor 转换为 NumPy 数组，并确保数据类型为 float32。FAISS 需要使用 NumPy 数组作为输入。
# embeddings_np = np.vstack([e.numpy() for e in embeddings]).astype('float32')
# print(embeddings_np)

# # 使用 FAISS 创建索引，这里使用的是 L2 距离(即欧式距离) 和点乘(归一化的向量点即cos相似度)
# index = faiss.IndexFlatL2(1024)  # 128 是向量的维度

# 将PyTorch张量转换为NumPy数组
embeddings_np = embeddings.numpy().astype('float32')

# 为查询文本生成嵌入向量
query_embedding = get_embedding([query], tokenizer, model).numpy().astype('float32')

# 初始化FAISS索引 - 假设embedding_dimension是你的嵌入向量维度
embedding_dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)  # 使用L2距离

# 将文本嵌入向量添加到索引中
index.add(embeddings_np)

# 执行搜索，查找与查询向量最相似的3个向量
D, I = index.search(query_embedding, 3)  # D: 距离, I: 索引

# 打印查询结果
print('查询文本:', query)
for i, idx in enumerate(I[0]):
    print(f"Rank {i+1}: Text: '{texts[idx]}', Distance: {D[0][i]}")