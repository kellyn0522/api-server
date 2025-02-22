# 환경변수 로딩
from dotenv import load_dotenv
load_dotenv()

# 모델 초기화
from langchain.chat_models import init_chat_model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain_core.messages import HumanMessage

# response = model.invoke([HumanMessage(content="Hi! I'm Bob")])
# print(response.content)

# 앞서 나의 이름을 말했지만, 다시 질문하면 모른다고 돌아옴.
# response = model.invoke([HumanMessage(content="What's my name?")])
# print(response.content)

# from langchain_core.messages import AIMessage

# response = model.invoke(
#     [
#         HumanMessage(content="Hi! I'm Bob"),
#         AIMessage(content="Hello Bob! How can I assist you today?"),
#         HumanMessage(content="What's my name?"),
#     ]
# )

# print(response.content)

# 챗봇 생성 
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 프롬프트 생성: 언어 번역
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

from langchain_core.messages import AIMessage, SystemMessage, trim_messages

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

trimmer.invoke(messages)

from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

# 커스텀 상태 정의: 언어 입력
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


# Define a new graph
workflow = StateGraph(state_schema=State)


# Define the function that calls the model. 인수는 우리가 정의한 State
# def call_model(state: State):
#     prompt = prompt_template.invoke(state)
#     response = model.invoke(prompt)
#     return {"messages": response}

# Define the function that calls the model. 인수는 우리가 정의한 State
def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": response}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# config = {"configurable": {"thread_id": "abc123"}}
# query = "Hi! I'm Bob."
# language = "Spanish"

config = {"configurable": {"thread_id": "abc678"}}
query = "What math problem did I ask?"
language = "English"

input_messages = [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()


# 챗봇 핑퐁-------------------
# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()  # output contains all messages in state

# query = "What's my name?"

# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()

# # 새로운 스레드로 멀티 유저 챗봇 만들기 가능 
# config = {"configurable": {"thread_id": "abc234"}}

# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()
# ---------------------------


# [현재 코드]
# 멀티 스레드를 지원하는 챗봇
# + 과거 대화 내용을 일정 길이(토큰 수)로 제한하는 트리머(trim_messages)를 적용해서 메모리 관리
# 
# [주요 기능]
# 1. 멀티 스레드 지원 : thread_id를 활용하여 여러 사용자(또는 대화 세션)를 관리
# 2. 대화 컨텍스트 유지 : MemorySaver()를 활용하여 과거 대화 기억
# 3. 대화 길이 조절 : trim_messages()를 사용해 토큰 제한(65)을 넘지 않도록 대화 내용 정리
# 4. 언어 설정 : 사용자가 설정한 언어로 답변 생성
