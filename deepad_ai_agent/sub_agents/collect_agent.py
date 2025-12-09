from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search, FunctionTool, AgentTool
from google.adk.models.google_llm import Gemini
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams

from google.genai import types
from google.genai.types import GenerateContentConfig

from ..prompts import COLLECT_AGENT_INSTR
from ..tools import tool_get_segment_lift_data, tool_get_naver_trend_data_api, tool_get_deepad_trend_data
from ..callbacks import before_tool_request, after_tool_response


model_collect_agent = Gemini(model_name="gemini-2.5-flash-preview-09-2025", retry_options=types.HttpRetryOptions(initial_delay=1, max_delay=3, attempts=3))
config_collect_agent_thinking = types.ThinkingConfig(include_thoughts=False, thinking_budget=0)



google_search_agent = Agent(
    model=model_collect_agent,
    name='GoogleSearchAgent',
    instruction="""
        당신은 제일 검색을 잘하는 에이전트입니다.
        구글 그라운딩 검색을 이용하여 요청한 키워드로 심도있는 자료를 검색하세요.
        **[출력 형식 강제 - 매우 중요]**
        검색 도구(`Google Search`)를 호출한 후, 해당 검색 결과를 요약하거나 변환하지 마십시오.
        대신, 검색 도구에서 반환된 원본 결과물 중 **가장 관련성이 높은 3개의 항목**에 대해 **제목(Title), URL(Link), 발췌 내용(Snippet)**을 포함하는 **JSON 리스트 형식**으로만 최종 결과물을 반환해야 합니다.

        **[JSON 형식 예시]**
        [
            {
                "title": "검색 결과의 제목",
                "url": "https://실제.검색.링크",
                "snippet": "검색 결과의 짧은 요약 내용"
            },
            {
                "title": "두 번째 결과 제목",
                "url": "https://두.번째.링크",
                "snippet": "두 번째 검색 결과 요약"
            }
        ]
    """,
    tools=[google_search]
)


collect_agent = Agent(
    model=model_collect_agent,
    name="collect_agent",
    description="분석 주제요청에 따라 데이터를 조회 및 수집하는 Agent",
    instruction=COLLECT_AGENT_INSTR,
    planner=BuiltInPlanner(
        thinking_config=config_collect_agent_thinking
    ),
    
    tools=[
        FunctionTool(tool_get_segment_lift_data),
        FunctionTool(tool_get_naver_trend_data_api),
        FunctionTool(tool_get_deepad_trend_data),
        AgentTool(agent=google_search_agent),
        MCPToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command="mcp",
                        args= [
                            "run",
                            # "/Volumes/DATA/Sources/Python/deepad_ai/py-mcp-naver-search-main/server.py"
                            "C:/ksson/Python/deepad_ai/py-mcp-naver-search-main/server.py"
                        ],
                        env= {
                            "NAVER_CLIENT_ID" : "nLBzU2Mg5vey3cmRa349",
                            "NAVER_CLIENT_SECRET" : "tdUinzCkI0"
                        },
                        encoding="utf-8",
                        encoding_error_handler="strict", 
                    )
                    , timeout=120
                )
            )
    ],
    generate_content_config=GenerateContentConfig(
        temperature=0.1, top_p=0.1
    ),
    before_tool_callback=before_tool_request,
    after_tool_callback=after_tool_response
)