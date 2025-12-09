from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search, FunctionTool, AgentTool
from google.adk.models.google_llm import Gemini
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams

from google.genai import types
from google.genai.types import GenerateContentConfig

from ..prompts import TOPIC_SELECTION_AGENT_INSTR
from ..tools import tool_get_naver_search_mcp, tool_get_naver_category_api, tool_get_deepad_category_data, tool_get_segment_info
from ..callbacks import before_tool_request, after_tool_response
from ..utils import logs_system

from enum import Enum
from typing import List
from pydantic import BaseModel, Field

model_topic_selection_agent = Gemini(model_name="gemini-2.5-flash-preview-09-2025", retry_options=types.HttpRetryOptions(initial_delay=1, max_delay=3, attempts=3))
config_topic_selection_agent_thinking = types.ThinkingConfig(include_thoughts=True, thinking_budget=1024)

google_search_agent = Agent(
    model=model_topic_selection_agent,
    name='GoogleSearchAgent',
    instruction="""
        당신은 제일 검색을 잘하는 에이전트입니다.
        구글 그라운딩 검색을 이용하여 요청한 키워드로 심도있는 자료를 검색하세요.
    """,
    tools=[google_search],
)

topic_selection_agent = Agent(
        model=model_topic_selection_agent,
        description="분석주제 선정 및 분석에 필요한 최소한의 정보 수집 후 분석계획을 수립하는 Agent",
        name="topic_selection_agent",
        instruction=TOPIC_SELECTION_AGENT_INSTR,
        generate_content_config=GenerateContentConfig(
            temperature=0.1,
            top_p=0.95
        ),
        planner=BuiltInPlanner(
            thinking_config=config_topic_selection_agent_thinking
        ),
        tools=[
            AgentTool(agent=google_search_agent),
            FunctionTool(tool_get_segment_info),
            FunctionTool(tool_get_naver_category_api),
            FunctionTool(tool_get_deepad_category_data),
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
        before_tool_callback=before_tool_request,
        after_tool_callback=after_tool_response

    )