# 랭체인 연결
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("Hi!"),
]

# print(model.invoke(messages))

# 개별 토큰 스트리밍
# for token in model.stream(messages):
#     print(token.content, end="|")

# 탬플릿 만드는 코드
from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Korean", "text": "hi!"})

response = model.invoke(prompt)
print(response.content)

