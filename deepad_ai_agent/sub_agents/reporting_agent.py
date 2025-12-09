from google.adk.agents import Agent, BaseAgent
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search, FunctionTool, AgentTool
from google.adk.models.google_llm import Gemini

from google.genai import types
from google.genai.types import GenerateContentConfig
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from typing import AsyncGenerator, List

from ..prompts import REPORTING_AGENT_INSTR
from ..tools import tool_create_html_report
from ..callbacks import after_tool_response


model_reporting_agent = Gemini(model_name="gemini-2.5-flash-preview-09-2025", retry_options=types.HttpRetryOptions(initial_delay=1, max_delay=3, attempts=3))
config_reporting_agent_thinking = types.ThinkingConfig(include_thoughts=True, thinking_budget=1024)

reporting_agent = Agent(
    model=model_reporting_agent,
    name="reporting_agent",
    description="분석 결과로 보고서 생성하는 Agent",
    instruction=REPORTING_AGENT_INSTR,
    planner=BuiltInPlanner(
        thinking_config=config_reporting_agent_thinking
    ),
    tools=[tool_create_html_report],
    generate_content_config=GenerateContentConfig(
        temperature=0.1, top_p=0.95
    ),
    after_tool_callback=after_tool_response

)
