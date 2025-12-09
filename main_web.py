from google.adk.runners import Runner, RunConfig
from google.adk.sessions import InMemorySessionService
from google.adk.sessions import DatabaseSessionService
from google.genai import types


from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

import sys, os, time, json, asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, field_validator, Field

from deepad_ai_agent.utils import logs_system, convert_markdown_to_html

from deepad_ai_agent.tools import tool_translate_ko, tool_get_naver_search_mcp
from langdetect import detect

from  deepad_ai_agent.agent import root_agent

#--------------- 추가부분 ----------------# 
import logging
logging.basicConfig(level=logging.ERROR) # Warnning 오류가 많이 발생하여 메세지 확인 어려운 문제로 ERROR 이상 로그만 표시하도록 설정.
#--------------- 추가부분 ----------------# 

DEBUG_MODE_YN = False

# 환경변수 로드
load_dotenv()
APP_NAME = "deepad_ai_research_app"

if sys.platform == "win32":
    # Python 3.8부터 ProactorEventLoop를 기본 이벤트 루프로 설정
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# 세션 서비스와 Agent는 전역으로 한 번만 초기화
session_service = InMemorySessionService()

# db_host_Public = os.getenv("DB_HOST_Public")
# db_name = os.getenv("DB_NAME")
# db_user = os.getenv("DB_USER")
# db_password = os.getenv("DB_PASS")
# db_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host_Public}:3306/{db_name}?charset=utf8"

# session_service = DatabaseSessionService(db_url=db_url, pool_recycle=200)

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    This is the recommended way to manage resources that need to be
    available for the lifetime of the application.
    """
    # Startup
    logs_system("Application startup...")
    
    app_state["runner"] = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service
    )
    # app_state["tool_naver_search_mcp"] = tool_naver_search_mcp
    logs_system("Runner and tools initialized. Application is ready.")
    
    yield
    
    # Shutdown
    logs_system("Application shutdown...")
    await app_state["tool_naver_search_mcp"].close()
    logs_system("MCP Tools closed gracefully.")

app = FastAPI(lifespan=lifespan)  

allowed_origins = [
    "null", 
    "https://dev-adm-deepad.lpoint.com",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )


async def init_mcp_tools():
    logs_system("MCP Tool 사용 준비중....")

    tool_naver_search_mcp = await tool_get_naver_search_mcp()

    logs_system("MCP Tools 사용 준비완료.")

    return tool_naver_search_mcp

class ChatRequest(BaseModel):
    query: str = Field(..., description="사용자 요청 텍스트")
    user_id: str = Field(default='anonymous', description="사용자 ID")
    session_id: str = Field(default='default_session', description="세션 ID")


# 실제 채팅 테스트를 하기 위한 html 페이지 표시
@app.get('/')
def get_chat_page():
    """ 채팅  페이지를 렌더링합니다. """
    return FileResponse('./templates/index_chat.html')


@app.post('/chat')
async def chat(chat_request: ChatRequest):
    """
    클라이언트(WEB)로부터 요청을 받아서 Agent 결과를 반환하는 함수
    """
    
    runner = app_state.get("runner")

    # 보고서 생성관련 Tool(사용자 정보 user_id, session_id 필요)
    # 사용자가 chat 호출 시 세션이 생성되어 해당 세션요청 정보받아서 reporting_agent tool에 보고서 저장 tool 추가로 저장.
    # 한번 추가되면 중복추가 안되게 tool 개수가 1개(gcs_upload)일 경우에만 추가

    return StreamingResponse(process_query(runner, chat_request), media_type="str")


async def process_query(runner: Runner, chat_request: ChatRequest):
    """
    Agent에게 요청 후 결과를 스트리밍으로 반환하는 함수
    yeild를 사용하여 응답을 스트리밍 형식으로 내보내는 형태
    """
    try:
        # 사용자에 대한 세션이 이미 존재하는지 확인합니다.
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=chat_request.user_id,
            session_id=chat_request.session_id,
        )

        # 세션이 존재하지 않으면 새로 생성합니다.
        if session is None:
            logs_system(f"Session '{chat_request.session_id}' not found for user '{chat_request.user_id}'. Creating a new one.")
            session = await session_service.create_session(
                app_name=APP_NAME,
                user_id=chat_request.user_id,
                session_id=chat_request.session_id,
                state={},  # 빈 상태로 세션 시작
            )
        
        # 실제 채팅진행 Process
        async for response_chunk in process_chat(runner=runner, chat_request=chat_request):
            yield response_chunk
       

    except Exception as e:
        if hasattr(e, 'code') and e.code == 429:
                logs_system("429 Rate Limit 오류가 발생하였습니다. query 수정 후 재시도 합니다.")
                chat_request.query = "계속해줘"
                raise e
        else:               
            print(f"스트리밍 중 오류 발생: {e}")
            err_msg = f"<p style='font-size: 11px; color: #bbb'>{str(e)}</p>"
            error_data = json.dumps({"event_type": "error", "content": str(e)}, ensure_ascii=False)
            error_data = json.dumps({"event_type": "error", "content": "처리 중 일시적으로 오류가 발생하였습니다. <br /><strong>\"다시 진행해줘\"</strong> 또는 <strong>\"계속 진행해줘\"</strong> 를 입력해 보시고 계속 발생할 경우 <strong>딥애드 관리자</strong>에게 문의해주세요<br />" + err_msg})
            yield f"data: {error_data}\n\n".encode('utf-8')

async def process_chat(runner: Runner, chat_request: ChatRequest):
     # LLM 행동 향상을 위해 "답변할 때 충분히 생각을 한 후 답변하세요" 멘트 추가
    user_query = chat_request.query# + "\n**Please think carefully before answering.**\n"
    content = types.Content(role='user', parts=[types.Part(text=user_query,)])

    run_config = RunConfig(streaming_mode="bidi")

    final_response_text = "Agent가 최종 응답을 생성하지 못했습니다."
    # 실제 LLM 결과 받아오는 부분
    async for event in runner.run_async(user_id=chat_request.user_id, session_id=chat_request.session_id, new_message=content, run_config=run_config):

        try:

            response_data = None
            # event_type : thinking / message / final_response
            event_type = "message" # 기본 이벤트 타입
            summary_text = "" # 요약 텍스트

            if event.usage_metadata != None:
                print(f"prompt_token_cnt: {event.usage_metadata.prompt_token_count} / prompt_thinking_cnt: {event.usage_metadata.thoughts_token_count} / total_token_cnt: {event.usage_metadata.total_token_count}")
                if event.usage_metadata.total_token_count > 800000 and event.author == "topic_selection_agent":
                     yield json.dumps({
                                "agent_name": event.author,
                                "event_type": "final_response",
                                "summary": "오류가 발생하였습니다",
                                "content": "토큰 초과로 더이상 채팅이 불가능합니다. 새로운 채팅을 클릭하여 진행해주세요."
                            }, ensure_ascii=False)

            if event.is_final_response():
                if event.content and event.content.parts:
                    event_type = "final_response"
                   

                    for part in event.content.parts:
                        
                        if (part.text != None):
                            # 디버깅 용 메세지 출력
                            logs_system("\n\n---------------------------- part(final_response) ---------------------------\n")
                            logs_system(str(part))
                            logs_system("\n-------------------------------------------------------------\n\n")

                            if (part.thought == True):
                                event_type = "thinking"
                                response_data = "(thinking) " + part.text
                            else:
                                event_type = "final_response"
                                response_data = part.text                                

                            if event_type == 'final_response':
                                # 답변이 영어로 되어 있을 때 번역 도구를 이용하여 번역
                                if (detect(response_data) == "en"):
                                    response_data = tool_translate_ko(response_data)

                            # 수집, 분석, 검증 에이전트의 경우 진행중으로 표시하기 위해 "message" type으로 전송
                            # if (event.author =="collect_agent" or event.author == "research_agent"):
                            #     event_type = "message"
                            
                            # 최종 응답내용은 markdown > html 형태로 변환하여 반환
                            final_html_data = convert_markdown_to_html(response_data)
                            # # 최종 변환 문자열이 바뀌는 문제 확인용
                            # logs_system("\n\n---------------------------- part(final_html_data) ---------------------------\n")
                            # logs_system(final_html_data)
                            # logs_system("\n-------------------------------------------------------------\n\n")

                            yield json.dumps({
                                "agent_name": event.author,
                                "event_type": event_type,
                                "summary": summary_text,
                                "content": final_html_data
                            }, ensure_ascii=False)
                elif event.error_code:
                    yield json.dumps({
                        "agent_name": event.author,
                        "event_type": "error",
                        "summary": "오류가 발생하였습니다.",
                        "content": event.error_message[0:1000] + ' ...'
                    })
            elif event.content.parts:
                for part in event.content.parts:
                    if (part.text != None):
                        logs_system("\n\n---------------------------- part(processing...) ---------------------------\n")
                        logs_system(str(part))
                        logs_system("\n-------------------------------------------------------------\n\n")
                        if (part.thought == True):
                            event_type = "thinking"
                            response_data = "(thinking) " + part.text
                        else:
                            event_type = "message"
                            response_data = part.text

                    elif part.function_call:
                        event_type = "function_call"                
                        process_data = f"function_call : {part.function_call.name}"
                        
                        if part.function_call.args:
                            process_data += f" args : {str(part.function_call.args)}"
                        
                        if DEBUG_MODE_YN == True:
                            response_data = process_data
                        else:
                            event_type = "message"
                            # function call 에 따라 summary 메세지 표시
                            if event.author == "root_agent":
                                if part.function_call.name == "transfer_to_agent":
                                    if part.function_call.args["agent_name"] == "topic_selection_agent":
                                        summary_text = "요청하신 주제에 대한 분석 계획을 수립 예정입니다..."
                                    elif part.function_call.args["agent_name"] == "research_workflow_agent":
                                        summary_text = "수립완료된 분석 계획에 따라 분석을 시작하겠습니다..."
                            elif event.author == "topic_selection_agent":
                                if part.function_call.name == "GoogleSearchAgent":
                                    summary_text = "요청 주제에 대한 분석계획 수립을 위해 기본정보를 조회하고 있습니다..."
                                elif part.function_call.name == "search_shop" or part.function_call.name == "tool_get_naver_category_api" or part.function_call.name == "tool_get_deepad_category_data":
                                    summary_text = "분석을 위해 요청 주제 해당하는 정확한 제품/카테고리 정보를 조회 중입니다..."
                                elif part.function_call.name == "tool_get_segment_info":
                                    summary_text = "요청 주제에 해당하는 세그먼트 정보를 조회하고 있습니다..."
                                else:
                                    summary_text = "분석 계획을 바탕으로 데이터 수집을 시작하겠습니다..."
                            elif event.author == "collect_agent":
                                if part.function_call.name == "tool_get_naver_trend_data_api":
                                    summary_text = "네이버 트렌드 데이터를 조회하고 있습니다..."
                                elif part.function_call.name == "tool_get_deepad_trend_data":
                                    summary_text = "딥애드 트렌드 데이터를 조회하고 있습니다..."
                                elif part.function_call.name == "tool_get_segment_lift_data":
                                    summary_text = "딥애드 사용자 세그먼트 연관성 분석 데이터를 조회하고 있습니다..."
                                elif part.function_call.name == "search_news" or part.function_call.name == "search_blog" or part.function_call.name == "search_cafe_article" or part.function_call.name == "search_webkr":
                                    summary_text = f"웹 데이터[{part.function_call.args["query"]} 외 다양한 키워드]를  수집 중입니다..."
                                elif part.function_call.name == "GoogleSearchAgent":
                                    summary_text = f"웹 데이터[{part.function_call.args["request"]} 외 다양한 키워드]를  수집 중입니다..."
                                else:
                                    summary_text = "수집된 데이터를 바탕으로 분석 진행 예정입니다...."
                            elif event.author == "research_agent":
                                if part.function_call.name == "tool_calculate_cmgr":
                                    summary_text = "월별성장률(CMGR)을 계산하고 있습니다..."
                                elif part.function_call.name == "tool_seasonal_decompose":
                                    summary_text = "시계열 분해 및 계절성 데이터를 조회 중입니다..."
                                else:
                                    summary_text = "최종 분석 결과를 바탕으로 보고서를 생성 예정입니다..."
                            elif event.author == "reporting_agent":
                                if part.function_call.name == "tool_create_html_report":
                                    summary_text = "보고서 생성도구를 이용하여 보고서 생성 중입니다..."
                                else:
                                    summary_text = "전체 분석 결과를 기준으로 보고서 생성 중입니다..."
                            
                            response_data = ""
                        
                       # logs_system(process_data)
                        logs_system(f"[{event.author}] {process_data}")
                        
                    elif part.function_response:
                        event_type = "function_response"
                        process_data = f"function_response: {part.function_response.name}"
                        if part.function_response.response:
                            # 결과 텍스트가 있을 경우 출력
                            resp = part.function_response.response
                            if ("result" in resp):
                                if (str(type(part.function_response.response.get("result"))) == "<class 'mcp.types.CallToolResult'>"):
                                    process_data += resp.get("result").content[0].text
                            else:
                                process_data += str(resp)
                        
                        if DEBUG_MODE_YN == True:
                            response_data = process_data
                        else:
                            if part.function_response.name == "search_webkr" or part.function_response.name == "search_news":
                                event_type = "message"
                                response_data = ""
                                search_results = json.loads(part.function_response.response.get("result").content[0].text)
                                search_results_cnt = len(search_results.get("items"))
                                for item in search_results.get("items"): 
                                    search_item = f"{item.get('title')}\n-------\n{item.get('link')}\n{item.get('description')}\n-------\n"
                                    response_data += search_item
                            elif part.function_response.name == "GoogleSearchAgent":
                                event_type = "message"
                                response_txt = part.function_response.response.get("result")

                                if response_txt.startswith("```json"):
                                    google_search_results = json.loads(response_txt.replace("```json", "").replace("```", "").strip())
                                    response_data = ""
                                    for item in google_search_results:
                                        search_item = f"{item.get('title')}\n-------\\n{item.get('url')}\n{item.get('snippet')}-------\\n"
                                        response_data += search_item
                                else:
                                    response_data = response_txt
                                
                            else:
                                response_data = None
                        
                        # logs_system(process_data)
                        logs_system(f"[{event.author}] function_response: {part.function_response.name}")

                        # 최종 응답내용은 markdown > html 형태로 변환하여 반환
                    if response_data != None:
                        yield json.dumps({
                            "agent_name": event.author,
                            "event_type": event_type,
                            "summary": summary_text,
                            "content": convert_markdown_to_html(response_data) 
                        }, ensure_ascii=False)
            else:
                logs_system("\n\n---------------------------- event(바로중단케이스...) ---------------------------\n")
                logs_system(str(event))
                logs_system("\n-------------------------------------------------------------\n\n")

        except Exception as e:
            # 주제선정 에이전트 또는 보고서 에이전트에서만 오류메세지 출력.
            if event.author == "topic_selection_agent" or event.author == "reporting_agent":
                print(f"스트리밍 중 오류 발생: {e}")
                err_msg = f"<p style='font-size: 11px; color: #bbb'>{str(e)}</p>"
                error_data = json.dumps({"event_type": "error", "content": str(e)}, ensure_ascii=False)
                error_data = json.dumps({"event_type": "error", "content": "처리 중 일시적으로 오류가 발생하였습니다. <br /><strong>\"다시 진행해줘\"</strong> 또는 <strong>\"계속 진행해줘\"</strong> 를 입력해 보시고 계속 발생할 경우 <strong>딥애드 관리자</strong>에게 문의해주세요<br />"+ err_msg})
                yield  error_data
    yield json.dumps({
        "event_type": "close",
        "content": "Agent가 최종 응답을 전송하여 연결이 닫혔습니다."
    }, ensure_ascii=False)