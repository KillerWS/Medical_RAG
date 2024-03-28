from zhipuai import ZhipuAI
client = ZhipuAI(api_key="9f916c050511132c0747dad352dd2ebc.yesz3ZEZ6wMYR1KW") # 填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-3-turbo", # 填写需要调用的模型名称
    messages=[
        # {"role": "user", "content": "作为一名营销专家，请为我的产品创作一个吸引人的slogan"},
        # {"role": "assistant", "content": "当然，为了创作一个吸引人的slogan，请告诉我一些关于您产品的信息"},
        # {"role": "user", "content": "智谱AI开放平台"},
        # {"role": "assistant", "content": "智启未来，谱绘无限一智谱AI，让创新触手可及!"},
        # {"role": "user", "content": "创造一个更精准、吸引人的slogan"}
        {"role": "user", "content": "我感冒了怎么办?用英文回答"}
    ],
)

print(response.choices[0].message)