import json
import os
import time
from typing import List
from main1 import send

from fastapi import FastAPI, Form, UploadFile, File
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse

app = FastAPI()
# 设置允许的源、方法和头部信息
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许的源列表
    allow_credentials=True,  # 是否允许携带凭证（如 cookie）
    allow_methods=["*"],  # 允许的方法列表，'*' 表示所有方法
    allow_headers=["*"],  # 允许的头部列表，'*' 表示所有头部
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


data = {
    "messages": [
        {"text": "Hello, World!", "timestamp": "2021-01-01T12:00:00"},
        {"text": "Hello, FastAPI!", "timestamp": "2021-01-02T12:00:01"},
        {"text": "Hello, FastAPI!", "timestamp": "2021-01-02T12:00:02"},
    ]
}


def generate_json_stream(data):
    # 分割JSON数据，并逐个发送
    for message in data["messages"]:
        json_str = "data: " + json.dumps(message) + "\n\n"
        yield json_str.encode("utf-8")
        time.sleep(1)  # 模拟延时


def generate_test_data():
    # for item in list('抱歉，您的需求暂时还无法满足，我只支持油气行业相关问答及油气行业文档的生成'):
    for item in list(
            "\n收到文件请求，共2个文件，开始解析文件\n\n第1个文件解析完成\n\n第1个文件入库成功\n\n第2个文件解析完成\n\n第2个文件入库成功\n开始生成模板\n\n选定格式文件：横4井钻井工程设计.docx\n\n开始生成模板文件，总共需要处理92段内容\n\n[status] 处理中，处理进度 0/92\n\n[status] 处理中，处理进度 1/92\n\n[status] 处理中，处理进度 2/92\n\n[status] 处理中，处理进度 3/92\n\n[status] 处理中，处理进度 4/92\n\n[status] 处理中，处理进度 5/92\n\n[status] 处理中，处理进度 6/92\n\n[status] 处理中，处理进度 7/92\n\n[status] 处理中，处理进度 8/92\n\n[status] 处理中，处理进度 9/92\n\n[status] 处理中，处理进度 10/92\n\n[status] 处理中，处理进度 11/92\n\n[status] 处理中，处理进度 12/92\n\n[status] 处理中，处理进度 13/92\n\n[status] 处理中，处理进度 14/92\n\n[status] 处理中，处理进度 15/92\n\n[status] 处理中，处理进度 16/92\n\n[status] 处理中，处理进度 17/92\n\n[status] 处理中，处理进度 18/92\n\n[status] 处理中，处理进度 19/92\n\n[status] 处理中，处理进度 20/92\n\n[status] 处理中，处理进度 21/92\n\n[status] 处理中，处理进度 22/92\n\n[status] 处理中，处理进度 23/92\n\n[status] 处理中，处理进度 24/92\n\n[status] 处理中，处理进度 25/92\n\n[status] 处理中，处理进度 26/92\n\n[status] 处理中，处理进度 27/92\n\n[status] 处理中，处理进度 28/92\n\n[status] 处理中，处理进度 29/92\n\n[status] 处理中，处理进度 30/92\n\n[status] 处理中，处理进度 31/92\n\n[status] 处理中，处理进度 32/92\n\n[status] 处理中，处理进度 33/92\n\n[status] 处理中，处理进度 34/92\n\n[status] 处理中，处理进度 35/92\n\n[status] 处理中，处理进度 36/92\n\n[status] 处理中，处理进度 37/92\n\n[status] 处理中，处理进度 38/92\n\n[status] 处理中，处理进度 39/92\n\n[status] 处理中，处理进度 40/92\n\n[status] 处理中，处理进度 41/92\n\n[status] 处理中，处理进度 42/92\n\n[status] 处理中，处理进度 43/92\n\n[status] 处理中，处理进度 44/92\n\n[status] 处理中，处理进度 45/92\n\n[status] 处理中，处理进度 46/92\n\n[status] 处理中，处理进度 47/92\n\n[status] 处理中，处理进度 48/92\n\n处理完成， 92/92\n\n开始渲染模板文件\n\n生成模板文件，文件地址:<download_file>template_agent_20240913163017.docx?module_id=3</download_file>\n\n生成模板文件，文件地址:<download_file>template_sllm_20240913163019.docx?module_id=3</download_file>\n\n生成模板文件，文件地址:<download_file>template_diff_20240913163021.docx?module_id=3</download_file>\n\n从datapool获取值进行模板填充\n\n 取数结果为 : \n\n\n ```json\n {\n    \"构造名称\": \"鄂尔多斯盆地伊陕斜坡\",\n    \"气候类型\": \"大陆性季风半干旱气候\",\n    \"年平均气温\": \"未提供具体数值\",\n    \"春季特征\": \"干旱多大风\",\n    \"夏季特征\": \"高温多雷雨\",\n    \"秋季特征\": \"凉爽而短促\",\n    \"冬季特征\": \"漫长且干旱\",\n    \"降雨特点\": \"雨热同季\",\n    \"日照条件\": \"充足\",\n    \"构造位置\": \"鄂尔多斯盆地伊陕斜坡\",\n    \"横坐标\": \"19350527.79m\",\n    \"纵坐标\": \"4148658.14m\",\n    \"地面海拔\": \"1238.46m\",\n    \"磁偏角\": \"-3°40′\",\n    \"井名\": \"tlx测试井名\",\n    \"钻探目的\": \"了解古生界二迭系石千峰组、石盒子组、山西组、太原组、石炭系本溪组储层发育及含油气情况；兼探中生界三迭系延长组油层\",\n    \"目的层位\": \"石千峰组,石盒子组,山西组,太原组,本溪组,马家沟组兼探延长组\",\n    \"设计井深\": \"3284m\",\n    \"完钻深度\": \"5520m\",\n    \"井别\": \"tlx测试井别\",\n    \"井型\": \"tlx测试井型\",\n    \"中生界油气层防喷工作\": \"做好中生界油气层的防喷工作, 特别是在甘泉、富县、黄龙等南部区域要注意浅层气，做好井控工作\",\n    \"周边注水区域注意事项\": \"在本井周围500m范围以内若有注水井或油井，要引起注意\",\n    \"邻井延507情况\": \"在延长组曾发生严重井漏，在刘家沟组至石千峰组发生渗漏\",\n    \"刘家沟组防漏堵漏工作\": \"特别注意做好表层钻进中的防漏堵漏工作\",\n    \"石千峰组～奥陶系的钻井液密度\": \"一般在1.06～1.07 g/cm3\",\n    \"钻井液密度控制\": \"小于1.03 g/cm3(三迭系以上), 1.06～1.07 g/cm3(石千峰组～奥陶系)\",\n    \"邻井延582情况\": \"在表层钻进过程中曾发生严重井漏\",\n    \"邻井三迭系以上地层钻井液密度\": \"小于1.03 g/cm3\",\n    \"古生界地层压力系数\": \"一般为0.868左右，钻井施工过程中应注意控制钻井液密度，保护好气层\",\n    \"应急措施\": \"严格按照含H2S气井井控规定和钻井安全操作规范处理\",\n    \"drilling_fluid_type\": \"water_based_drilling_fluid\",\n    \"施工要求\": \"加强对H2S气体的录井检测及防范\",\n    \"表格名称\": \"横4井钻井液基本性能要求表\",\n    \"目标\": \"确保人身及钻井设备的安全\",\n    \"全井钻井液体系要求\": \"1.4.8\",\n    \"钻井液使用情况\": \"完钻\",\n    \"危险气体\": \"H2S\",\n    \"年平均降雨量\": \"未提供具体数值\",\n    \"无霜期\": \"146d\",\n    \"夏秋季节灾害\": \"山洪、山体滑坡\",\n    \"春季天气现象\": \"多沙尘暴\",\n    \"第四系含水性\": \"较强\",\n    \"春秋季风向\": \"西北风\",\n    \"潜水位深度\": \"较浅\",\n    \"target_formation\": \"石千峰组, 石盒子组, 山西组, 太原组, 本溪组, 马家沟组 兼探 延长组\",\n    \"地表物质\": \"第四系未固结成岩的松散黄色粘土\",\n    \"交通状况\": \"距主干公路较远，交通不太方便\",\n    \"通讯条件\": \"基本上能满足要求\",\n    \"本溪组高压层防喷措施\": \"进入气层后，尤其是本溪组属局部高压层，要特别注意做好防喷工作\",\n    \"本井H2S含量预测\": \"1.4.5\",\n    \"设计依据\": \"横4井钻井地质设计\",\n    \"表格编号\": \"表1-4\",\n    \"location\": \"陕西省榆林市横山县石湾镇碾盘湾北西约594m处\",\n    \"地形特征\": \"沟壑梁峁发育，地形起伏较大\",\n    \"地理位置描述\": \"黄土高原腹地\",\n    \"Magnetic_Declination\": \"-3°40′\",\n    \"ground_elevation\": \"1238.46m\",\n    \"预测气层位置\": \"未提及具体值\",\n    \"完井方法\": \"全套管射孔完井\",\n    \"地层压力预测\": \"本井目的层\",\n    \"设计地层\": \"未提及具体值\",\n    \"本溪组高压层注意事项\": \"进入气层后，尤其是本溪组属局部高压层，本井要特别注意做好防喷工作,特别要注意防范由于井漏诱发井喷事故\",\n    \"邻井钻井液密度使用情况\": \"三迭系以上地层一般使用钻井液密度小于1.03 g/cm3，石千峰组～奥陶系的钻井液密度一般在1.06～1.07 g/cm3\",\n    \"井口坐标.横\": 19350527.79,\n    \"井口坐标.纵\": 4148658.14,\n    \"geological_stratification\": \"not specified\",\n    \"oil_gas_water_layers\": \"not specified\",\n    \"magnetic_deviation\": \"-3°40′\",\n    \"周边注水情况\": \"可能存在注水区域\",\n    \"H2S存在\": \"是\",\n    \"防漏堵漏重点地层\": \"刘家沟组\",\n    \"地层压力系数\": \"0.868左右\",\n    \"钻井液密度范围\": \"1.03 g/cm3以下至1.06～1.07 g/cm3\",\n    \"地理位置\": \"浙江省杭州市\",\n    \"任务\": \"本井目的层地层压力预测\",\n    \"防碰要求\": \"1.4.10\",\n    \"刘家沟组漏失情况\": \"邻井延582和延507在钻进过程中曾发生严重井漏，本井应特别注意做好表层钻进中的防漏堵漏工作\",\n    \"防喷工作重点地层\": \"中生界油气层, 本溪组\",\n    \"井控注意事项\": \"注意浅层气, 防范由于井漏诱发井喷事故\",\n    \"防喷工作重点\": \"中生界油气层和浅层气防范\",\n    \"特殊注意事项\": \"防范井漏诱发井喷事故\",\n    \"特殊地质风险\": \"刘家沟组漏失严重, 本溪组属局部高压层\",\n    \"页眉内容\": \"特雷西文档页眉\",\n    \"井号\": \"tlxtlxtxl\",\n    \"日期\": \"2024-09-13\",\n    \"海拔值\": \"8848\"\n} \n```\n\n文件生成，地址：<download_file>result_agent_20240913163017.docx?module_id=3</download_file>\n\n文件生成，地址：<download_file>result_sllm_20240913163019.docx?module_id=3</download_file>\n\n文件生成，地址：<download_file>result_diff_20240913163021.docx?module_id=3</download_file>\n"):
        output = {
            "question_item": "你好啊",
            "question_time_id": "1724812755208",
            "modelType": "PetroLLAMA",
            "dataSource": "Redis",
            "needFileNum": 5,
            "maxTokens": 256,
            "top_k": 10,
            "top_p": 0.95,
            "repetitionPenalty": 1.0,
            "temperature": 0.1,
            "rate": 6.0,
            "topic_id": "1724812755208",
            "conver_id": "1724812755208",
            "module_id": "3",
            "report_template": "",
            "reference_report": "",
            "question": {
                "item": "你好啊",
                "time_id": "1724812755208"
            },
            "ID": 0,
            "answer": {
                "item": [
                    {
                        "order": 1,
                        "source": "",
                        "type": "text",
                        "data": item
                    }
                ],
                "timestamp": 1727164582.9313162
            }
        }
        yield f'data: {json.dumps(output, ensure_ascii=False)}\n\n'
        time.sleep(0.1)
    yield 'data: [DONE]\n\n'


@app.get("/generate")
async def stream_data():
    return StreamingResponse(generate_test_data(), media_type="text/event-stream")
    # return StreamingResponse(generate_json_stream(data), media_type="text/event-stream")


@app.post("/all_drop_down_data")
async def all_drop_down_data():
    return {
        "topic_id": "default",
        "model_type": [
            "PetroLLAMA"
        ],
        "data_source": [
            "runimage"
        ],
        "save_file_type": [
            "txt",
            "docx",
            "xlsx"
        ],
        "top_k": 10,
        "top_p": 0.95,
        "repetitionpenalty": 1,
        "temperature": 0.1,
        "max_tokens": 256,
        "rate": "0.6",
        "report_template": [
            "周长_模板_20240823115433",
            "南部勘探区_模板_差异化结果_20240830063945_on_延2203井钻井工程设计",
            "神木_模板_agent结果_20240829195841_on_延1919-1井钻井工程设计",
            "南部勘探区_模板_agent结果_20240830063945_on_延2203井钻井工程设计",
            "清涧_模板_差异化结果_20240830055413_on_延1825井钻井工程设计",
            "延149井区_模板_单节点结果_20240909200536_on_长36-4井钻井工程设计"
        ],
        "reference_report": [
            "tag20240828",
            "tag0913"
        ]
    }


@app.post("/get_topic_key")
async def get_topic_key():
    return [
        {
            "topic_id": "1725331395013",
            "dataSource": "Redis",
            "modelType": "PetroLLAMA",
            "topic_name": "\u94bb\u4e95\u901a\u5e38\u90fd\u7528\u4ec0\u4e48\u94bb\u4e95\u6db2"
        },
        {
            "topic_id": "1725342652106",
            "dataSource": "Redis",
            "modelType": "PetroLLAMA",
            "topic_name": "\u94bb\u4e95\u6db2\u662f\u7528\u6765\u505a\u4ec0\u4e48\u7684"
        },
        {
            "topic_id": "1725002220800",
            "dataSource": "Redis",
            "modelType": "PetroLLAMA",
            "topic_name": "\u5468\u957f\u4e95\u533a\u6709\u591a\u5c11\u53e3\u4e95"
        },
        {
            "topic_id": "1725111667699",
            "dataSource": "Redis",
            "modelType": "PetroLLAMA",
            "topic_name": "\u94bb\u4e95\u7528\u4e86\u4ec0\u4e48\u94bb\u4e95\u6db2"
        },
        {
            "topic_id": "1724841946304",
            "dataSource": "Redis",
            "modelType": "PetroLLAMA",
            "topic_name": "\u7ed9\u4f60\u4e2a\u6a21\u677f\uff0c\u5e2e\u6211\u751f\u6210\u4e00\u4e2a\u62a5\u544a"
        },
        {
            "topic_id": "1724900786343",
            "dataSource": "Redis",
            "modelType": "PetroLLAMA",
            "topic_name": "\u94bb\u4e95\u4f5c\u4e1a\u5b8c\u6210\u540e\u5e94\u8be5\u600e\u4e48\u7ef4\u62a4\u73af\u5883"
        },
        {
            "topic_id": "1724896964076",
            "dataSource": "Redis",
            "modelType": "PetroLLAMA",
            "topic_name": "\u94bb\u4e95\u8fc7\u7a0b\u4e2d\u7684\u8425\u5730\u5b89\u5168\u65b9\u9762\u53ef\u4ee5\u5c55\u5f00\u8bf4\u8bf4\u5417"
        },
        {
            "topic_id": "1724841054529",
            "dataSource": "Redis",
            "modelType": "PetroLLAMA",
            "topic_name": "\u7ed9\u6211\u8bf4\u4e2d\u6587\uff0c\u94bb\u4e95\u5de5\u7a0b\u6709\u54ea\u4e9b\u73af\u8282"
        }
    ]


@app.post("/get_topic_history")
async def get_topic_history():
    return [
        {
            "question_item": "\u5e2e\u6211\u751f\u6210\u4e951\u7684\u94bb\u4e95\u62a5\u544a",
            "question_time_id": "1726803903500",
            "modelType": "PetroLLAMA",
            "dataSource": "runimage",
            "needFileNum": 5,
            "maxTokens": 256,
            "top_k": 10,
            "top_p": 0.95,
            "repetitionPenalty": 1.0,
            "temperature": 0.1,
            "rate": 6.0,
            "topic_id": "1726803903500",
            "conver_id": "1726803903500",
            "module_id": "3",
            "report_template": "",
            "reference_report": "",
            "domain": "\u5468\u957f",
            "question": {
                "item": "\u5e2e\u6211\u751f\u6210\u4e951\u7684\u94bb\u4e95\u62a5\u544a",
                "time_id": "1726803903500"
            },
            "ID": 0,
            "answer": {
                "item": [
                    {
                        "ID": 0,
                        "type": "text",
                        "succ": True,
                        "stop": True,
                        "source": "",
                        "data": {
                            "items": [
                                {
                                    "data": "\u5f53\u524d\u652f\u6301\"\u8bd5\u6c14\u5730\u8d28\u8bbe\u8ba1\", \"\u8bd5\u6c14\u5de5\u7a0b\u8bbe\u8ba1\", \"\u94bb\u4e95\u5730\u8d28\u8bbe\u8ba1\", \"\u94bb\u4e95\u5de5\u7a0b\u8bbe\u8ba1\"\u56db\u79cd\u62a5\u544a\u751f\u6210\uff0c\u8bf7\u660e\u786e\u60a8\u8981\u751f\u6210\u7684\u62a5\u544a\u7c7b\u578b"
                                }
                            ]
                        }
                    }
                ],
                "timeID": 1726803943.0802631
            }
        },
        {
            "question_item": "\u5e2e\u6211\u751f\u6210\u8bd5\u6c14\u5730\u8d28\u8bbe\u8ba1\u62a5\u544a",
            "question_time_id": "1726803958449",
            "modelType": "PetroLLAMA",
            "dataSource": "runimage",
            "needFileNum": 5,
            "maxTokens": 256,
            "top_k": 10,
            "top_p": 0.95,
            "repetitionPenalty": 1.0,
            "temperature": 0.1,
            "rate": 6.0,
            "topic_id": "1726803903500",
            "conver_id": "1726803958449",
            "module_id": "3",
            "report_template": "",
            "reference_report": "",
            "domain": "\u5468\u957f",
            "question": {
                "item": "\u5e2e\u6211\u751f\u6210\u8bd5\u6c14\u5730\u8d28\u8bbe\u8ba1\u62a5\u544a",
                "time_id": "1726803958449"
            },
            "ID": 1,
            "answer": {
                "item": [
                    {
                        "ID": 0,
                        "type": "text",
                        "succ": True,
                        "stop": True,
                        "source": "",
                        "data": {
                            "items": [
                                {
                                    "data": "\u8bf7\u660e\u786e\u60a8\u8981\u751f\u6210\u62a5\u544a\u7684\u94bb\u4e95"
                                }
                            ]
                        }
                    }
                ],
                "timeID": 1726803998.0346587
            }
        },
        {
            "question_item": "\u8bf7\u5e2e\u6211\u751f\u6210\u4e953\u7684\u94bb\u4e95\u5730\u8d28\u8bbe\u8ba1\u62a5\u544a",
            "question_time_id": "1726803979227",
            "modelType": "PetroLLAMA",
            "dataSource": "runimage",
            "needFileNum": 5,
            "maxTokens": 256,
            "top_k": 10,
            "top_p": 0.95,
            "repetitionPenalty": 1.0,
            "temperature": 0.1,
            "rate": 6.0,
            "topic_id": "1726803903500",
            "conver_id": "1726803979227",
            "module_id": "3",
            "report_template": "",
            "reference_report": "",
            "domain": "\u5468\u957f",
            "question": {
                "item": "\u8bf7\u5e2e\u6211\u751f\u6210\u4e953\u7684\u94bb\u4e95\u5730\u8d28\u8bbe\u8ba1\u62a5\u544a",
                "time_id": "1726803979227"
            },
            "ID": 2,
            "answer": {
                "item": [
                    {
                        "ID": 0,
                        "type": "text",
                        "succ": True,
                        "stop": True,
                        "source": "",
                        "data": {
                            "items": [
                                {
                                    "data": "\n\u89e3\u6790\u6a21\u677f\u6587\u4ef6\n\n\u4ece\u6570\u636e\u5e93\u4e2d\u53d6\u503c\n\n \u53d6\u6570\u7ed3\u679c\u4e3a : \n```json\n {\n    \"\u9875\u7709\u5185\u5bb9\": \"\u7279\u96f7\u897f\u6587\u6863\u9875\u7709\",\n    \"\u4e95\u540d\": \"tlx\u6d4b\u8bd5\u4e95\u540d\",\n    \"\u4e95\u53f7\": \"tlxtlxtxl\",\n    \"\u4e95\u522b\": \"tlx\u6d4b\u8bd5\u4e95\u522b\",\n    \"\u4e95\u578b\": \"tlx\u6d4b\u8bd5\u4e95\u578b\",\n    \"\u65e5\u671f\": \"2024-09-19\",\n    \"\u5730\u7406\u4f4d\u7f6e\": \"\u6d59\u6c5f\u7701\u676d\u5dde\u5e02\",\n    \"\u6d77\u62d4\u503c\": \"8848\"\n} \n```\n\n\n[status] \u5904\u7406\u8fdb\u5ea6 1/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 2/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 3/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 4/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 5/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 6/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 7/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 8/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 9/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 10/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 11/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 12/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 13/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 14/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 15/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 16/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 17/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 18/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 19/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 20/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 21/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 22/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 23/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 24/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 25/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 26/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 27/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 28/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 29/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 30/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 31/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 32/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 33/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 34/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 35/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 36/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 37/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 38/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 39/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 40/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 41/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 42/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 43/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 44/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 45/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 46/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 47/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 48/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 49/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 50/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 51/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 52/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 53/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 54/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 55/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 56/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 57/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 58/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 59/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 60/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 61/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 62/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 63/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 64/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 64/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 65/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 66/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 67/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 68/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 69/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 70/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 71/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 72/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 73/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 74/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 75/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 76/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 77/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 78/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 79/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 80/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 81/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 82/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 83/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 84/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 85/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 86/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 87/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 88/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 89/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 90/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 91/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 92/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 93/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 94/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 95/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 96/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 97/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 98/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 99/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 100/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 101/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 102/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 103/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 104/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 105/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 106/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 107/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 108/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 109/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 110/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 111/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 112/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 113/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 114/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 115/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 116/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 117/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 118/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 119/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 120/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 121/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 122/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 123/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 124/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 125/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 126/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 127/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 128/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 129/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 130/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 131/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 132/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 133/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 134/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 135/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 136/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 137/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 138/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 139/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 140/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 141/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 142/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 143/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 144/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 145/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 146/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 147/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 148/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 149/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 150/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 151/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 152/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 153/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 154/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 155/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 156/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 157/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 158/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 159/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 160/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 161/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 162/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 163/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 164/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 165/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 166/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 167/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 168/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 169/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 170/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 171/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 172/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 173/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 174/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 175/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 176/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 177/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 178/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 179/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 180/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 181/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 182/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 183/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 184/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 185/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 186/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 187/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 188/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 189/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 190/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 191/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 192/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 193/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 194/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 195/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 196/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 197/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 198/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 199/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 200/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 201/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 202/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 203/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 204/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 205/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 206/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 207/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 208/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 209/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 210/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 211/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 212/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 213/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 214/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 215/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 216/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 217/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 218/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 219/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 220/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 221/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 222/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 223/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 224/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 225/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 226/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 227/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 228/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 229/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 230/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 231/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 232/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 233/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 234/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 235/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 236/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 237/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 238/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 239/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 240/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 241/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 242/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 243/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 244/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 245/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 246/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 247/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 248/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 249/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 250/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 251/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 252/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 253/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 254/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 255/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 256/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 257/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 258/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 259/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 260/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 261/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 262/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 263/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 264/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 265/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 266/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 267/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 268/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 269/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 270/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 271/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 272/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 273/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 274/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 275/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 276/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 277/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 278/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 279/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 280/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 281/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 282/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 283/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 284/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 285/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 286/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 287/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 288/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 289/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 290/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 291/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 292/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 293/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 294/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 295/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 296/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 297/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 298/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 299/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 300/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 301/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 302/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 303/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 304/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 305/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 306/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 307/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 308/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 309/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 310/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 311/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 312/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 313/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 314/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 315/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 316/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 317/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 318/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 319/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 320/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 320/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 321/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 322/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 323/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 324/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 325/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 326/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 327/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 328/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 329/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 329/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 330/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 331/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 332/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 333/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 334/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 335/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 336/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 337/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 338/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 339/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 340/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 341/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 342/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 343/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 344/345\n\n[status] \u5904\u7406\u8fdb\u5ea6 345/345\n\n\u751f\u6210\u6587\u6863\uff1a<download_file>\u5468\u957f_\u6a21\u677f_agent\u7ed3\u679c_20240920111239_on_\u54348\u94bb\u4e95\u5730\u8d28\u8bbe\u8ba1.docx?module_id=3</download_file>\n"
                                }
                            ]
                        }
                    }
                ],
                "timeID": 1726804018.813121
            }
        }
    ]
    #     [
    #     {
    #         "question_item": "钻井中都会用到哪些钻井液",
    #         "question_time_id": "1726211058036",
    #         "modelType": "PetroLLAMA",
    #         "dataSource": "runimage",
    #         "needFileNum": 5,
    #         "maxTokens": 256,
    #         "top_k": 10,
    #         "top_p": 0.95,
    #         "repetitionPenalty": 1,
    #         "temperature": 0.1,
    #         "rate": 6,
    #         "topic_id": "1726211058036",
    #         "conver_id": "1726211058036",
    #         "module_id": "3",
    #         "report_template": "周长_模板_20240823115433",
    #         "reference_report": "tag0913",
    #         "question": {
    #             "item": "钻井中都会用到哪些钻井液",
    #             "time_id": "1726211058036"
    #         },
    #         "ID": 0,
    #         "answer": {
    #             "item": [
    #                 {
    #                     "ID": 0,
    #                     "type": "text",
    #                     "succ": True,
    #                     "stop": True,
    #                     "source": "",
    #                     "data": {
    #                         "items": [
    #                             {
    #                                 "data": "在钻井过程中，会使用到两种主要类型的钻井液。首先，在延长组至石千峰组的钻井阶段，使用的钻井液是低固相聚合物钻井液体系。然后，从石千峰50开始直到完钻，采用的是三磺钻井液体系。"
    #                             }
    #                         ]
    #                     }
    #                 }
    #             ],
    #             "timeID": "1726211082.5088212"
    #         }
    #     },
    #     {
    #         "question_item": "帮我生成文档",
    #         "question_time_id": "1726211520524",
    #         "modelType": "PetroLLAMA",
    #         "dataSource": "runimage",
    #         "needFileNum": 5,
    #         "maxTokens": 256,
    #         "top_k": 10,
    #         "top_p": 0.95,
    #         "repetitionPenalty": 1,
    #         "temperature": 0.1,
    #         "rate": 6,
    #         "topic_id": "1726211520524",
    #         "conver_id": "1726211520524",
    #         "module_id": "3",
    #         "report_template": "",
    #         "reference_report": "tag0913",
    #         "question": {
    #             "item": "帮我生成文档",
    #             "time_id": "1726211520524"
    #         },
    #         "ID": 0,
    #         "answer": {
    #             "item": [
    #                 {
    #                     "ID": 0,
    #                     "type": "text",
    #                     "succ": True,
    #                     "stop": True,
    #                     "source": "",
    #                     "data": {
    #                         "items": [
    #                             {
    #                                 "data": "\n收到文件请求，共2个文件，开始解析文件\n\n第1个文件解析完成\n\n第1个文件入库成功\n\n第2个文件解析完成\n\n第2个文件入库成功\n开始生成模板\n\n选定格式文件：横4井钻井工程设计.docx\n\n开始生成模板文件，总共需要处理92段内容\n\n[status] 处理中，处理进度 0/92\n\n[status] 处理中，处理进度 1/92\n\n[status] 处理中，处理进度 2/92\n\n[status] 处理中，处理进度 3/92\n\n[status] 处理中，处理进度 4/92\n\n[status] 处理中，处理进度 5/92\n\n[status] 处理中，处理进度 6/92\n\n[status] 处理中，处理进度 7/92\n\n[status] 处理中，处理进度 8/92\n\n[status] 处理中，处理进度 9/92\n\n[status] 处理中，处理进度 10/92\n\n[status] 处理中，处理进度 11/92\n\n[status] 处理中，处理进度 12/92\n\n[status] 处理中，处理进度 13/92\n\n[status] 处理中，处理进度 14/92\n\n[status] 处理中，处理进度 15/92\n\n[status] 处理中，处理进度 16/92\n\n[status] 处理中，处理进度 17/92\n\n[status] 处理中，处理进度 18/92\n\n[status] 处理中，处理进度 19/92\n\n[status] 处理中，处理进度 20/92\n\n[status] 处理中，处理进度 21/92\n\n[status] 处理中，处理进度 22/92\n\n[status] 处理中，处理进度 23/92\n\n[status] 处理中，处理进度 24/92\n\n[status] 处理中，处理进度 25/92\n\n[status] 处理中，处理进度 26/92\n\n[status] 处理中，处理进度 27/92\n\n[status] 处理中，处理进度 28/92\n\n[status] 处理中，处理进度 29/92\n\n[status] 处理中，处理进度 30/92\n\n[status] 处理中，处理进度 31/92\n\n[status] 处理中，处理进度 32/92\n\n[status] 处理中，处理进度 33/92\n\n[status] 处理中，处理进度 34/92\n\n[status] 处理中，处理进度 35/92\n\n[status] 处理中，处理进度 36/92\n\n[status] 处理中，处理进度 37/92\n\n[status] 处理中，处理进度 38/92\n\n[status] 处理中，处理进度 39/92\n\n[status] 处理中，处理进度 40/92\n\n[status] 处理中，处理进度 41/92\n\n[status] 处理中，处理进度 42/92\n\n[status] 处理中，处理进度 43/92\n\n[status] 处理中，处理进度 44/92\n\n[status] 处理中，处理进度 45/92\n\n[status] 处理中，处理进度 46/92\n\n[status] 处理中，处理进度 47/92\n\n[status] 处理中，处理进度 48/92\n\n处理完成， 92/92\n\n开始渲染模板文件\n\n生成模板文件，文件地址:<download_file>template_agent_20240913163017.docx?module_id=3</download_file>\n\n生成模板文件，文件地址:<download_file>template_sllm_20240913163019.docx?module_id=3</download_file>\n\n生成模板文件，文件地址:<download_file>template_diff_20240913163021.docx?module_id=3</download_file>\n\n从datapool获取值进行模板填充\n\n 取数结果为 : \n\n\n ```json\n {\n    \"构造名称\": \"鄂尔多斯盆地伊陕斜坡\",\n    \"气候类型\": \"大陆性季风半干旱气候\",\n    \"年平均气温\": \"未提供具体数值\",\n    \"春季特征\": \"干旱多大风\",\n    \"夏季特征\": \"高温多雷雨\",\n    \"秋季特征\": \"凉爽而短促\",\n    \"冬季特征\": \"漫长且干旱\",\n    \"降雨特点\": \"雨热同季\",\n    \"日照条件\": \"充足\",\n    \"构造位置\": \"鄂尔多斯盆地伊陕斜坡\",\n    \"横坐标\": \"19350527.79m\",\n    \"纵坐标\": \"4148658.14m\",\n    \"地面海拔\": \"1238.46m\",\n    \"磁偏角\": \"-3°40′\",\n    \"井名\": \"tlx测试井名\",\n    \"钻探目的\": \"了解古生界二迭系石千峰组、石盒子组、山西组、太原组、石炭系本溪组储层发育及含油气情况；兼探中生界三迭系延长组油层\",\n    \"目的层位\": \"石千峰组,石盒子组,山西组,太原组,本溪组,马家沟组兼探延长组\",\n    \"设计井深\": \"3284m\",\n    \"完钻深度\": \"5520m\",\n    \"井别\": \"tlx测试井别\",\n    \"井型\": \"tlx测试井型\",\n    \"中生界油气层防喷工作\": \"做好中生界油气层的防喷工作, 特别是在甘泉、富县、黄龙等南部区域要注意浅层气，做好井控工作\",\n    \"周边注水区域注意事项\": \"在本井周围500m范围以内若有注水井或油井，要引起注意\",\n    \"邻井延507情况\": \"在延长组曾发生严重井漏，在刘家沟组至石千峰组发生渗漏\",\n    \"刘家沟组防漏堵漏工作\": \"特别注意做好表层钻进中的防漏堵漏工作\",\n    \"石千峰组～奥陶系的钻井液密度\": \"一般在1.06～1.07 g/cm3\",\n    \"钻井液密度控制\": \"小于1.03 g/cm3(三迭系以上), 1.06～1.07 g/cm3(石千峰组～奥陶系)\",\n    \"邻井延582情况\": \"在表层钻进过程中曾发生严重井漏\",\n    \"邻井三迭系以上地层钻井液密度\": \"小于1.03 g/cm3\",\n    \"古生界地层压力系数\": \"一般为0.868左右，钻井施工过程中应注意控制钻井液密度，保护好气层\",\n    \"应急措施\": \"严格按照含H2S气井井控规定和钻井安全操作规范处理\",\n    \"drilling_fluid_type\": \"water_based_drilling_fluid\",\n    \"施工要求\": \"加强对H2S气体的录井检测及防范\",\n    \"表格名称\": \"横4井钻井液基本性能要求表\",\n    \"目标\": \"确保人身及钻井设备的安全\",\n    \"全井钻井液体系要求\": \"1.4.8\",\n    \"钻井液使用情况\": \"完钻\",\n    \"危险气体\": \"H2S\",\n    \"年平均降雨量\": \"未提供具体数值\",\n    \"无霜期\": \"146d\",\n    \"夏秋季节灾害\": \"山洪、山体滑坡\",\n    \"春季天气现象\": \"多沙尘暴\",\n    \"第四系含水性\": \"较强\",\n    \"春秋季风向\": \"西北风\",\n    \"潜水位深度\": \"较浅\",\n    \"target_formation\": \"石千峰组, 石盒子组, 山西组, 太原组, 本溪组, 马家沟组 兼探 延长组\",\n    \"地表物质\": \"第四系未固结成岩的松散黄色粘土\",\n    \"交通状况\": \"距主干公路较远，交通不太方便\",\n    \"通讯条件\": \"基本上能满足要求\",\n    \"本溪组高压层防喷措施\": \"进入气层后，尤其是本溪组属局部高压层，要特别注意做好防喷工作\",\n    \"本井H2S含量预测\": \"1.4.5\",\n    \"设计依据\": \"横4井钻井地质设计\",\n    \"表格编号\": \"表1-4\",\n    \"location\": \"陕西省榆林市横山县石湾镇碾盘湾北西约594m处\",\n    \"地形特征\": \"沟壑梁峁发育，地形起伏较大\",\n    \"地理位置描述\": \"黄土高原腹地\",\n    \"Magnetic_Declination\": \"-3°40′\",\n    \"ground_elevation\": \"1238.46m\",\n    \"预测气层位置\": \"未提及具体值\",\n    \"完井方法\": \"全套管射孔完井\",\n    \"地层压力预测\": \"本井目的层\",\n    \"设计地层\": \"未提及具体值\",\n    \"本溪组高压层注意事项\": \"进入气层后，尤其是本溪组属局部高压层，本井要特别注意做好防喷工作,特别要注意防范由于井漏诱发井喷事故\",\n    \"邻井钻井液密度使用情况\": \"三迭系以上地层一般使用钻井液密度小于1.03 g/cm3，石千峰组～奥陶系的钻井液密度一般在1.06～1.07 g/cm3\",\n    \"井口坐标.横\": 19350527.79,\n    \"井口坐标.纵\": 4148658.14,\n    \"geological_stratification\": \"not specified\",\n    \"oil_gas_water_layers\": \"not specified\",\n    \"magnetic_deviation\": \"-3°40′\",\n    \"周边注水情况\": \"可能存在注水区域\",\n    \"H2S存在\": \"是\",\n    \"防漏堵漏重点地层\": \"刘家沟组\",\n    \"地层压力系数\": \"0.868左右\",\n    \"钻井液密度范围\": \"1.03 g/cm3以下至1.06～1.07 g/cm3\",\n    \"地理位置\": \"浙江省杭州市\",\n    \"任务\": \"本井目的层地层压力预测\",\n    \"防碰要求\": \"1.4.10\",\n    \"刘家沟组漏失情况\": \"邻井延582和延507在钻进过程中曾发生严重井漏，本井应特别注意做好表层钻进中的防漏堵漏工作\",\n    \"防喷工作重点地层\": \"中生界油气层, 本溪组\",\n    \"井控注意事项\": \"注意浅层气, 防范由于井漏诱发井喷事故\",\n    \"防喷工作重点\": \"中生界油气层和浅层气防范\",\n    \"特殊注意事项\": \"防范井漏诱发井喷事故\",\n    \"特殊地质风险\": \"刘家沟组漏失严重, 本溪组属局部高压层\",\n    \"页眉内容\": \"特雷西文档页眉\",\n    \"井号\": \"tlxtlxtxl\",\n    \"日期\": \"2024-09-13\",\n    \"海拔值\": \"8848\"\n} \n```\n\n文件生成，地址：<download_file>result_agent_20240913163017.docx?module_id=3</download_file>\n\n文件生成，地址：<download_file>result_sllm_20240913163019.docx?module_id=3</download_file>\n\n文件生成，地址：<download_file>result_diff_20240913163021.docx?module_id=3</download_file>\n"
    # 
    #                             }
    #                         ]
    #                     }
    #                 }
    #             ],
    #             "timeID": "1726211545.3560627"
    #         }
    #     },
    #     {
    #         "question_item": "\u57fa\u4e8e\u7ed9\u5b9a\u7684\u6a21\u677f\u5e2e\u6211\u751f\u6210\u4e00\u4e2a\u6587\u6863",
    #         "question_time_id": "1726222513473",
    #         "modelType": "PetroLLAMA",
    #         "dataSource": "runimage",
    #         "needFileNum": 5,
    #         "maxTokens": 256,
    #         "top_k": 10,
    #         "top_p": 0.95,
    #         "repetitionPenalty": 1.0,
    #         "temperature": 0.1,
    #         "rate": 6.0,
    #         "topic_id": "1726222513473",
    #         "conver_id": "1726222513473",
    #         "module_id": "3",
    #         "report_template": "\u5ef6969\u4e95\u533a_\u8bd5\u6c14\u5730\u8d28\u8bbe\u8ba1_\u6a21\u677f_\u5dee\u5f02\u5316\u7ed3\u679c_20240912183743_on_\u6a2a3-4\u4e95\u9a6c\u4e9422+\u9a6c\u4e9414+\u9a6c\u4e9413+\u9a6c\u4e9412+\u5c712\u5c42\u8bd5\u91c7\u5730\u8d28\u8bbe\u8ba1",
    #         "reference_report": "",
    #         "question": {
    #             "item": "\u57fa\u4e8e\u7ed9\u5b9a\u7684\u6a21\u677f\u5e2e\u6211\u751f\u6210\u4e00\u4e2a\u6587\u6863",
    #             "time_id": "1726222513473"
    #         },
    #         "ID": 0,
    #         "answer": {
    #             "item": [
    #                 {
    #                     "ID": 0,
    #                     "type": "text",
    #                     "succ": True,
    #                     "stop": True,
    #                     "source": "",
    #                     "data": {
    #                         "items": [
    #                             {
    #                                 "data": "[status] 处理进度 0/137\n\n[status] 处理进度 1/137\n\n[status] 处理进度 2/137\n\n[status] 处理进度 3/137\n\n[status] 处理进度 4/137\n\n[status] 处理进度 5/137\n\n[status] 处理进度 6/137\n\n[status] 处理进度 7/137\n\n[status] 处理进度 8/137\n\n[status] 处理进度 9/137\n\n[status] 处理进度 10/137\n\n[status] 处理进度 11/137\n\n[status] 处理进度 12/137\n\n[status] 处理进度 13/137\n\n[status] 处理进度 14/137\n\n[status] 处理进度 15/137\n\n[status] 处理进度 16/137\n\n[status] 处理进度 17/137\n\n[status] 处理进度 18/137\n\n[status] 处理进度 19/137\n\n[status] 处理进度 20/137\n\n[status] 处理进度 21/137\n\n[status] 处理进度 22/137\n\n[status] 处理进度 23/137\n\n[status] 处理进度 24/137\n\n[status] 处理进度 25/137\n\n[status] 处理进度 26/137\n\n[status] 处理进度 27/137\n\n[status] 处理进度 28/137\n\n[status] 处理进度 29/137\n\n[status] 处理进度 30/137\n\n[status] 处理进度 31/137\n\n[status] 处理进度 32/137\n\n[status] 处理进度 33/137\n\n[status] 处理进度 34/137\n\n[status] 处理进度 35/137\n\n[status] 处理进度 36/137\n\n[status] 处理进度 37/137\n\n[status] 处理进度 38/137\n\n[status] 处理进度 39/137\n\n[status] 处理进度 40/137\n\n[status] 处理进度 41/137\n\n[status] 处理进度 42/137\n\n[status] 处理进度 43/137\n\n[status] 处理进度 44/137\n\n[status] 处理进度 45/137\n\n[status] 处理进度 46/137\n\n[status] 处理进度 47/137\n\n[status] 处理进度 48/137\n\n[status] 处理进度 49/137\n\n[status] 处理进度 50/137\n\n[status] 处理进度 51/137\n\n[status] 处理进度 52/137\n\n[status] 处理进度 53/137\n\n[status] 处理进度 54/137\n\n[status] 处理进度 55/137\n\n[status] 处理进度 56/137\n\n[status] 处理进度 57/137\n\n[status] 处理进度 58/137\n\n[status] 处理进度 59/137\n\n[status] 处理进度 60/137\n\n[status] 处理进度 61/137\n\n[status] 处理进度 62/137\n\n[status] 处理进度 63/137\n\n[status] 处理进度 64/137\n\n[status] 处理进度 65/137\n\n[status] 处理进度 66/137\n\n[status] 处理进度 67/137\n\n[status] 处理进度 68/137\n\n[status] 处理进度 69/137\n\n[status] 处理进度 70/137\n\n[status] 处理进度 71/137\n\n[status] 处理进度 72/137\n\n[status] 处理进度 73/137\n\n[status] 处理进度 74/137\n\n[status] 处理进度 75/137\n\n[status] 处理进度 76/137\n\n[status] 处理进度 77/137\n\n[status] 处理进度 78/137\n\n[status] 处理进度 79/137\n\n[status] 处理进度 80/137\n\n[status] 处理进度 81/137\n\n[status] 处理进度 82/137\n\n[status] 处理进度 83/137\n\n[status] 处理进度 84/137\n\n[status] 处理进度 85/137\n\n[status] 处理进度 86/137\n\n[status] 处理进度 87/137\n\n[status] 处理进度 88/137\n\n[status] 处理进度 89/137\n\n[status] 处理进度 90/137\n\n[status] 处理进度 91/137\n\n[status] 处理进度 92/137\n\n[status] 处理进度 93/137\n\n[status] 处理进度 94/137\n\n[status] 处理进度 95/137\n\n[status] 处理进度 96/137\n\n[status] 处理进度 97/137\n\n[status] 处理进度 98/137\n\n[status] 处理进度 99/137\n\n[status] 处理进度 100/137\n\n[status] 处理进度 101/137\n\n[status] 处理进度 102/137\n\n[status] 处理进度 103/137\n\n[status] 处理进度 104/137\n\n[status] 处理进度 105/137\n\n[status] 处理进度 106/137\n\n[status] 处理进度 107/137\n\n[status] 处理进度 108/137\n\n[status] 处理进度 109/137\n\n[status] 处理进度 110/137\n\n[status] 处理进度 111/137\n\n[status] 处理进度 112/137\n\n[status] 处理进度 113/137\n\n[status] 处理进度 114/137\n\n[status] 处理进度 115/137\n\n[status] 处理进度 116/137\n\n[status] 处理进度 117/137\n\n[status] 处理进度 118/137\n\n[status] 处理进度 119/137\n\n[status] 处理进度 120/137\n\n[status] 处理进度 121/137\n\n[status] 处理进度 122/137\n\n[status] 处理进度 123/137\n\n[status] 处理进度 124/137\n\n[status] 处理进度 125/137\n\n[status] 处理进度 126/137\n\n[status] 处理进度 127/137\n\n[status] 处理进度 128/137\n\n[status] 处理进度 129/137\n\n[status] 处理进度 130/137\n\n[status] 处理进度 131/137\n\n[status] 处理进度 132/137\n\n[status] 处理进度 133/137\n\n[status] 处理进度 134/137\n\n[status] 处理进度 135/137\n\n[status] 处理进度 136/137\n\n[status] 处理进度 137/137\n\n生成文档：<download_file>延969井区_试气地质设计_模板_差异化结果_20240912183743_on_横3-4井马五22+马五14+马五13+马五12+山2层试采地质设计.docx?module_id=3</download_file>\n"
    #                             }
    #                         ]
    #                     }
    #                 }
    #             ],
    #             "timeID": 1726222538.4876022
    #         }
    #     }
    # ]


@app.post("/upload_file_list/")
async def upload_files(
        module_id: str = Form(...),
        user_id: str = Form(...),
        param_name: str = Form(...),
        tag: str = Form(...),
        files: List[UploadFile] = File(...), ):
    print(len(files))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6114)
