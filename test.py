#加载环境变量
import os

import google.generativeai as genai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

from IPython.display import Markdown
import textwrap

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


model = genai.GenerativeModel(model_name = "gemini-pro")
prompt_parts = [
    "你跟文心一言有什么关系？","我感冒了怎么办?"
]
response = model.generate_content(prompt_parts)
print(response.text)
# to_markdown(response.text)