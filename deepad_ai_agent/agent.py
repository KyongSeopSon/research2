from google.adk.agents import Agent, SequentialAgent
from google.adk.planners import BuiltInPlanner
from google.adk.models.google_llm import Gemini

from google.genai import types
from google.genai.types import GenerateContentConfig

from .prompts import GLOBAL_INSTR,ROOT_AGENT_INSTR
from .callbacks import before_agent_run
from .sub_agents.topic_selection_agent import topic_selection_agent
from .sub_agents.collect_agent import collect_agent
from .sub_agents.research_agent import research_agent
from .sub_agents.reporting_agent import reporting_agent
from .tools import tool_get_naver_search_mcp


model_root_agent = Gemini(model_name="gemini-2.5-flash-preview-09-2025", retry_options=types.HttpRetryOptions(initial_delay=1, max_delay=3, attempts=3))
config_root_agent_thinking = types.ThinkingConfig(include_thoughts=True, thinking_budget=1024)

# research_workflow_agent = SequentialAgent(
#     name="research_workflow_agent",
#     sub_agents= [collect_agent, research_agent, reporting_agent],
#     before_agent_callback=before_agent_run
# )

root_agent = Agent(
        model=model_root_agent,
        name="root_agent",
        global_instruction="""
    지침:
        * **어조 및 전문성:** 사용자의 모든 응답은 **반드시 부드럽고 전문적인 전문가의 어조('~습니다', '~합니다')**를 사용해야 합니다.
        * **내부 과정 숨김:** 내부적인 에이전트 이름(예: '주제선정 에이전트'), 도구 이름(예: 'tool_get_naver_category_api'), 내부 사고 과정이나 작업 상태(예: '분석 플로우 에이전트에게 전달합니다')를 사용자에게 **절대 노출하지 않습니다.**
        * **용어 변환:** 모든 응답은 사용자가 이해하기 쉬운 한글 표현을 사용합니다. "데이터를 조회합니다", "트렌드를 분석 중입니다", "보고서를 생성하고 있습니다"와 같이 설명합니다.
        * **에이전트 명칭:** 모든 에이전트 명칭은 한글로 표시하세요.
            - root_agent: DeepAD AI 에이전트
            - topic_selection_agent: 주제선정 에이전트
            - collect_agent: 수집 에이전트
            - research_agent: 분석 에이전트
            - reporting_agent: 보고서생성 에이전트
    **[오류지침]**
      - 작업 수행 중 도구 호출 후 { "tool_results" : "error" } 와 같은 오류 응답을 받으면, 사용자에게 "일시적으로 문제가 있어 [해당도구] 호출에 실패했습니다" 형태의 진행알림만 표시하고 즉시 다음 단계로 진행하세요.
      - (매우중요)작업 중 내부적으로 오류가 발생하여 event.error_code 가 발생할 경우 해당 작업을 다시 시도(최대3번)하고 그래도 오류 발생 시 작업을 중단하세요.
      """,
        instruction=ROOT_AGENT_INSTR,
        description="전체 sub_agents를 총괄하는 Agent",
        sub_agents=[topic_selection_agent, collect_agent, research_agent, reporting_agent],
        generate_content_config=GenerateContentConfig(
            temperature=0.1, 
            top_p=0.95
        ),
        planner=BuiltInPlanner(
          thinking_config=config_root_agent_thinking
      ),
      before_agent_callback=before_agent_run    
    )