from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search, FunctionTool
from google.adk.models.google_llm import Gemini

from google.genai import types
from google.genai.types import GenerateContentConfig

from .prompts import TOPIC_SELECTION_AGENT_INSTR
from .tools import tool_get_naver_search_mcp, tool_get_naver_category_api, tool_get_deepad_category_data, tool_get_segment_info
from .callbacks import after_tool_response, after_model_response

model_topic_selection_agent = Gemini(model_name="gemini-2.5-flash", retry_options=types.HttpRetryOptions(initial_delay=1, attempts=2))
config_topic_selection_agent_thinking = types.ThinkingConfig(include_thoughts=False, thinking_budget=0)

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
            google_search,
            FunctionTool(tool_get_segment_info),
            FunctionTool(tool_get_naver_category_api),
            FunctionTool(tool_get_deepad_category_data),
            tool_get_naver_search_mcp
        ],
        after_tool_callback=after_tool_response,
        after_model_callback=after_model_response

    )