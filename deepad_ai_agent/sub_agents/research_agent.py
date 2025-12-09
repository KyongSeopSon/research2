from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search, FunctionTool, AgentTool
from google.adk.models.google_llm import Gemini

from google.genai import types
from google.genai.types import GenerateContentConfig

from ..prompts import RESEARCH_AGENT_INSTR
from ..tools import tool_calculate_cmgr, tool_seasonal_decompose, tool_get_segment_lift_data
from ..callbacks import after_tool_response


model_research_agent = Gemini(model_name="gemini-2.5-pro", retry_options=types.HttpRetryOptions(initial_delay=1, max_delay=3, attempts=3))
config_research_agent_thinking = types.ThinkingConfig(include_thoughts=True, thinking_budget=1024)


research_agent = Agent(
    model=model_research_agent,
    name="research_agent",
    description="조회 또는 수집된 데이터를 분석하고 인사이트를 도출하는 Agent",
    instruction=RESEARCH_AGENT_INSTR,
    planner=BuiltInPlanner(
        thinking_config=config_research_agent_thinking
    ),
    generate_content_config=GenerateContentConfig(
        temperature=0.5, top_p=1
    ),
    after_tool_callback=after_tool_response,
    tools = [ 
        tool_get_segment_lift_data,
        tool_calculate_cmgr,
        tool_seasonal_decompose
    ]
    
)