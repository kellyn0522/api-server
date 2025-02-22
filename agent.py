# [검색 기능을 포함한 지능형 Q&A 챗봇]
#
# ReAct(Reasoning+Acting) 에이전트 생성 코드 -> 외부 도구(검색 API)를 활용할 수 있는 챗봇
# 
# [주요 기능]
# 1. Tavily 검색 API 연동 (TavilySearchResults)
#   - 웹 검색 기능을 추가해서 실시간 정보를 가져올 수 있도록 함
#   - max_result=2 -> 검색 결과를 최대 2개까지만 반환
# 2. LLM 기반 에이전트 생성
#   - create_react_agent(model, tools, checkpointer=memory)
#   - LLM을 사용하면서, 검색 같은 외부 도구를 활용할 수 있는 에이전트 만듦
# 3. 멀티 스레드 대화 관리 (MemorySaver)
#   - thread_id를 사용해 여러 사용자 세션을 독립적으로 관리 가능
# 4. ReAct 프레임워크 적용
#   - 질문 받고 -> 생각 -> 검색(필요시) -> 답변 생성
#   - 단순히 프롬프트에 따라 답변하는게 아니라, 필요할 경우 검색을 먼저 수행하고, 그 결과를 바탕으로 응답

from dotenv import load_dotenv
load_dotenv() 

from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=2)
# search_results = search.invoke("what is the weather in SF")
# print(search_results)
# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4", model_provider="openai")

from langchain_core.messages import HumanMessage

from langgraph.prebuilt import create_react_agent

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

agent_executor = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

# 챗봇 인사
response = agent_executor.invoke(
    {"messages": [HumanMessage(content="hi im bob!")]}, config
)

print(response["messages"][-1].content)

# 이전 대화를 기억하는지 확인
response = agent_executor.invoke(
    {"messages": [HumanMessage(content="whats my name?")]}, config
)

print(response["messages"][-1].content)

# LLM 자체는 기억을 유지하지 않으므로 MemorySaver()로 상태 관리 
# 따라서, 현재는 전의 기억 유지 가능