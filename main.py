from langchain.schema.runnable import RunnableMap
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch

#创建model
model = ChatGoogleGenerativeAI(model="gemini-pro")

#创建prompt模板
template = """Answer the question a full sentence, 
based only on the following context:
{context}
Question: {question}
"""

#由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

#生成embedding模型
bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

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

# #创建检索器
bge_retriever = vectordb.as_retriever(search_kwargs={"k": 1})

#创建chain
chain = RunnableMap({
    "context": lambda x: bge_retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | StrOutputParser()

response = chain.invoke({"question":"人从哪里来？"})
print(response)