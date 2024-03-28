from langchain.document_loaders import TextLoader
loader = TextLoader("../txt/麦t.txt")
documents = loader.load()
