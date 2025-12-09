from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import get_user_choice
from google.adk.tools.base_tool import BaseTool
from google.adk.agents import LlmAgent
from google.adk.models.llm_response import LlmResponse
from google.adk.models.llm_request import LlmRequest
from datetime import datetime
from typing import Optional, Dict, Any
from google.genai.types import Content

import re
import json
import time

def before_agent_run(callback_context: CallbackContext) -> Optional[Content]:
    """A callback function to initialize and manage the conversation context."""
    agent_name = callback_context.agent_name
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    current_state = callback_context.state

    current_state["current_time"] = current_time

    if "naver_category_results" not in current_state:
        current_state["naver_category_results"] = []
    
    if "deepad_category_results" not in current_state:
        current_state["deepad_category_results"] = []
    
    if "analysis_plan" not in current_state:
        current_state["analysis_plan"] = ""
    
    if "naver_trend_data_results" not in current_state:
        current_state["naver_trend_data_results"] = []
    
    if "deepad_trend_data_results" not in current_state:
        current_state["deepad_trend_data_results"] = []
    
    if "collected_search_results" not in current_state:
        current_state["collected_search_results"] = []
    
    if "naver_trend_cmgr" not in current_state:
        current_state["naver_trend_cmgr"] = []
    
    if "deepad_trend_cmgr" not in current_state:
        current_state["deepad_trend_cmgr"] = []
    
    if "analysis_results" not in current_state:
        current_state["analysis_results"] = ""
    
    if "report_file_url" not in current_state:
        current_state["report_file_url"] = ""

    return None

import random
def before_tool_request(tool, args, tool_context):
    agent_name = tool_context.agent_name
    tool_name = tool.name
    # 검색 API 호출 시 429 오류 발생 방지
    if tool.name == "GoogleSearchAgent" or tool.name == "search_shop" or tool.name == "search_news" or tool.name == "search_blog" or tool.name == "search_cafe_article":
        sec = random.randint(1,9) * 0.1
        time.sleep(sec)

def after_tool_response(
    tool, args, tool_context: ToolContext, tool_response
) -> None:
    current_state = tool_context.state

    # 네이버 카테고리 조회 결과 저장
    if tool.name == "tool_get_naver_category_api":
        json_naver_category_results =  current_state["naver_category_results"]
        json_naver_category_results.append(tool_response)    
        current_state["naver_category_results"] = json_naver_category_results
    
    # 딥애드 카테고리 조회 결과 저장
    if tool.name == "tool_get_deepad_category_data":
        json_deepad_category_results = current_state["deepad_category_results"]
        json_response = json.loads(tool_response)
        if (len(json_response) > 0):
            for ct in json_response:
                json_deepad_category_results.append(ct)
            current_state["deepad_category_results"] = json_deepad_category_results

    # 네이버 트렌드 데이터 저장
    if tool.name == "tool_get_naver_trend_data_api":
        json_naver_trend_results = current_state["naver_trend_data_results"]

        trend_type = "naver_trend"
        title = tool_response["results"][0]["title"]
        data = tool_response["results"][0]["data"]

        trend_data_results = {
            "trend_type": trend_type,
            "title": title,
            "data": data
        }

        json_naver_trend_results.append(trend_data_results)
        current_state["naver_trend_data_results"] = json_naver_trend_results
    
    # 딥애드 트렌드 데이터 저장
    if tool.name == "tool_get_deepad_trend_data":
        json_deepad_trend_results = current_state["deepad_trend_data_results"]
        json_deepad_trend_results.append(tool_response)
        current_state["deepad_trend_data_results"] =json_deepad_trend_results

    # 네이버 검색 데이터 저장
    if tool.name == "search_news" or tool.name == "search_blog" or tool.name == "search_cafe_article" or tool.name == "search_webkr":
        json_collected_search_results = current_state["collected_search_results"]

        response_text = tool_response.content[0].text

        if "error" not in response_text:
            naver_search_results = json.loads(response_text)

            for item in naver_search_results.get('items'):
                json_search_data = {
                    "type": tool.name,
                    "title": item.get('title'),
                    "link": item.get('link')
                }

                json_collected_search_results.append(json_search_data)
            
            current_state["collected_search_results"] = json_collected_search_results
    
    if tool.name == "GoogleSearchAgent":
        json_collected_search_results = current_state["collected_search_results"]

        if tool_response.startswith("```json"):
            google_search_results = json.loads(tool_response.replace("```json", "").replace("```", "").strip())

            for item in google_search_results:
                json_search_data = {
                    "type": tool.name,
                    "title": item.get("title"),
                    "link": item.get("url")
                }

                json_collected_search_results.append(json_search_data)

            current_state["collected_search_results"] = json_collected_search_results
            
            




        current_state["collected_search_results"] = json_collected_search_results

    
    # 월별 성장률(CMGR) 저장
    if tool.name == "tool_calculate_cmgr":
        naver_trend_cmgr = current_state["naver_trend_cmgr"]
        deepad_trend_cmgr = current_state["deepad_trend_cmgr"]

        for response in tool_response:
            if response["trend_type"] == "naver_trend":
                naver_trend_cmgr.append(response)
            elif response["trend_type"] == "deepad_trend":
                deepad_trend_cmgr.append(response)
        
        current_state["naver_trend_cmgr"] = naver_trend_cmgr
        current_state["deepad_trend_cmgr"] = deepad_trend_cmgr
    
    # 시계열 데이터 저장
    if tool.name == "tool_seasonal_decompose":
        json_naver_trend_results = current_state["naver_trend_data_results"]
        json_deepad_trend_results = current_state["deepad_trend_data_results"]
        json_naver_trend_list = []
        json_deepad_trend_list = []
        for resp in tool_response:
            if "trend_type" in resp:
                if resp["trend_type"] == "naver_trend":
                    json_naver_trend_list.append(resp)
                elif resp["trend_type"] == "deepad_trend":
                    json_deepad_trend_list.append(resp)
        
        json_naver_trend_results = json_naver_trend_list
        json_deepad_trend_results = json_deepad_trend_list
    
    # 보고서 업로드 완료 후 링크 URL 저장
    if tool.name == "tool_create_html_report":
        current_state["report_file_url"] = tool_response
        print(f"current_state['report_file_url'] : {tool_response}")

    # 디버깅용
    # if tool.name != "tool_get_deepad_category_data":
    #     p(rint(f"tool_response\n--------------------------\n{tool_response}\n---------------------------\n")