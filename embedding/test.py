import numpy as np
import faiss

from transformers import BertTokenizer, BertModel
import torch

# # 初始化模型和分词器
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# # 函数：将文本转换为嵌入向量
# def text_to_embedding(text, tokenizer, model):
#     # 对文本进行编码
#     encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
#     # 获取模型输出
#     with torch.no_grad():
#         output = model(**encoded_input)
#     # 获取嵌入向量（例如，取最后一层的平均）
#     embedding = output.last_hidden_state.mean(dim=1)
#     return embedding.numpy()

# # 示例文本
# text = "Hello, world!"
# # 生成嵌入向量
# embedding = text_to_embedding(text, tokenizer, model)


texts = [
    "这是文本1。",
    "这是文本2。",
    "这是文本3。",
    "这是文本4。",
    "这是文本5。",
    "这是文本6。",
    "这是文本7。",
    "这是文本8。",
    "这是文本9。",
    "这是文本10。"
]

query_text = "查询文本。"

# 使用随机数据模拟文本嵌入
np.random.seed(42)  # 确保结果可复现
embeddings = np.random.rand(len(texts), 512).astype('float32')  # 生成样本数据的嵌入向量
query_embedding = np.random.rand(1, 512).astype('float32')  # 生成查询数据的嵌入向量

# 初始化 FAISS 索引
dimension = 512  # 嵌入向量的维度
index = faiss.IndexFlatL2(dimension)  # 使用 L2 距离
index.add(embeddings)  # 向索引中添加样本数据的嵌入向量

# 进行搜索，找到与查询最相似的3个样本
k = 3  # 返回最相似的3个结果
D, I = index.search(query_embedding, k)  # D 是距离数组，I 是索引数组

# 打印出最相似的3个样本
print("最相似的3个样本的索引：", I[0])
print("对应的距离：", D[0])
for i in I[0]:
    print(texts[i])
