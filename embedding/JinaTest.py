# 文本向量化过程

from langchain.embeddings import HuggingFaceBgeEmbeddings

from transformers import AutoTokenizer, AutoModel
import torch

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
for embedding in embeddings:
    print( + embedding)

query = "Langage model Google"

# # 为文本列表生成嵌入向量
# docs = DocumentArray([Document(text=text, embedding=get_embedding(text, model, tokenizer)) for text in texts])

# # 为查询生成嵌入向量
# query_doc = Document(embedding=get_embedding(query, model, tokenizer))

# # 使用向量相似度在文档数组中寻找最相似的前3个文本
# docs.match(query_doc, limit=3)

# # 打印最相似的文本
# for match in query_doc.matches:
#     print(f"Score: {match.scores['cosine'].value}, Text: {match.text}")

