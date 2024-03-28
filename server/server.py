from flask import Flask, request, jsonify
from flask_cors import CORS
from zhipuai import ZhipuAI
import os
app = Flask(__name__)
CORS(app)  # 允许所有来源的 CORS 请求

# 从环境变量加载API_KEY
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

ZHUPUAI_API_KEY = os.getenv('ZHUPUAI_API_KEY')

client = ZhipuAI(api_key="9f916c050511132c0747dad352dd2ebc.yesz3ZEZ6wMYR1KW")  # 请填写您自己的APIKey

# 初始化一个空列表来保存对话历史

conversation_history = []
def getResponse(user_message, model_name, round):
    
    global conversation_history # 声明 conversation_history 为全局变量

    messages = {"role": "user", "content": user_message}
     # 将用户的最新消息添加到对话历史中
    conversation_history.append(messages)

    # 这里使用 messages 列表作为输入调用 ZhipuAI 的 chat.completions.create 方法
    response = client.chat.completions.create(
        model=model_name,  # 根据需要调用的模型进行调整
        messages=conversation_history
    )
    
    
    assistant_message_content = '' #确保值不为空
    if response and response.choices:
        # 确保访问字典中的 'content' 字段
        assistant_message_content  = response.choices[0].message.content
        # print(assistant_message_content)
        assistant_message = {"role": "assistant", "content": assistant_message_content}
        
        # 将生成的回复也添加到对话历史中，并保持长度不超过10
        conversation_history.append(assistant_message)

        if len(conversation_history) == round*2:
            conversation_history = [];
        
        print(conversation_history)
        
        return assistant_message_content


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    print(data.get("messages"))
  
    user_message = data.get('messages', '')  # 如果没有 'messages'，默认为空字符串
    
    # response = getResponse(user_message, "glm-4", 3)
    response = getResponse(user_message, "glm-3-turbo", 3)
    if response is not None:
        return jsonify({"message": response}), 200
    else:
        return jsonify({"message": "无法生成回复"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
