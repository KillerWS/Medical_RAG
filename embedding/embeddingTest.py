from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch

bge_embeddings = HuggingFaceBgeEmbeddings(model_name="../models/BAAI/bge-large-zh-v1.5")

# create vector DB   
vectordb = DocArrayInMemorySearch.from_texts(
    ["青蛙是食草动物",
     "人是由恐龙进化而来的。",
     "熊猫喜欢吃天鹅肉。",
     "1+1=5",
     "2+2=8",
     "3+3=9",
    "Gemini Pro is a Large Language Model was made by GoogleDeepMind",
     "A Language model is trained by predicting the next token"
    ],
    embedding=bge_embeddings 
)
 
#创建retriever
bge_retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# 搜索查询
query = "Langage model Google"


# 获取查询的嵌入表示
# query_embedding = bge_embeddings.get_embedding(query)
# print(query_embedding)

# 让它每次只返回1条最相关的文档：search_kwargs={"k": 1}
#Test cases search_kwargs={"k": 1}
reponse = bge_retriever.get_relevant_documents(query)
print(reponse)
# print(reponse)
