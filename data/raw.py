import requests
from langchain.document_loaders import TextLoader

url = "https://myzhengyuan.com/post/93853.html"
res = requests.get(url)
with open("93853.html", "w") as f:
    f.write(res.text)

loader = TextLoader('./93853.html')
documents = loader.load()