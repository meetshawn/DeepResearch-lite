# -*- coding: utf-8 -*-
"""
（使用网络搜索和LLM） - 带FastAPI前端

此脚本自动化生成行业分析报告的过程，通过以下步骤：
1.  接收用户关于特定行业的初始查询。
2.  使用 LLM 将查询分解为可搜索的子查询（规划）。
3.  使用 Bochaai API 对这些子查询执行网络搜索。
4.  整合和去重搜索结果。
5.  使用 LLM 评估收集到的信息，并在需要时建议进一步的子查询（反思）。
6.  重复搜索和反思步骤，直到达到最大迭代次数。
7.  （可选）对收集到的文本片段执行基本数据分析（关键词频率、数字提取）。
8.  使用流式 LLM（Deepseek）基于所有收集的信息合成最终报告，并适当地引用来源。

新增FastAPI前端提供交互界面。
"""

import requests
import json
from openai import OpenAI
import os
import time
import re
from urllib.parse import urlparse
from collections import Counter
import logging
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
import uuid
from pathlib import Path
from dotenv import load_dotenv

# --- 配置 ---

# API 密钥:
# 加载.env文件
load_dotenv()

# 读取配置
SEARCH_API_KEY = os.getenv("BOCHAAI_API_KEY")
LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY")
LLM_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FastAPI 应用设置 ---
app = FastAPI(title="行业分析报告生成器")

# 设置静态文件和模板目录
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 创建报告存储目录
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# --- 行业配置 ---
INDUSTRY_CONFIGS = {
      "deepResearch": {
        "name": "深度研究",
        "filename_prefix": "deepResearch_",
        "llm_system_prompt_assistant": "你是一位资深分析师。请仔细遵循指示并以要求的格式回答。",
        "llm_system_prompt_synthesizer": "你是你是一位资深分析师，负责根据收集的信息生成客观的信息分析。请严格遵循指示，并使用 `[来源: URL]` 格式引用来源。",
        "analyzer_keywords": [],
        "plan_prompt_template": """
            你是一个问题拆解专家，将用户的主要问题分解为具体的、可搜索的子问题列表，以全面分析该主题。请严格按照以下框架将用户输入的问题拆解为结构化的子问题树。

            1. 问题类型判定（20字以内）
            2. 核心维度识别（3-5个关键分析轴）
            3. 分层子问题树（至少3级结构）
            4. 边界确认问题（2-3个验证性问题）
            
            拆解方法论

            采用「逻辑金字塔」原则：
            1. 横向：MECE原则（相互独立，完全穷尽）
            2. 纵向：5Why递进法
            3. 边界：5W2H覆盖

            请严格按照以下 JSON 格式输出，以下仅示例仅仅作为参考，不要包含任何额外的解释或评论：
            {{
              "subqueries": [
                "子问题1: 用户问题的核心关键词和关键概念是什么？",
                "子问题2: 这个问题涉及的时间范围或时效性要求是什么？",
                "子问题3: 问题涉及的地理范围或特定区域是哪些？",
                "子问题4: 需要分析的主要影响因素有哪些？",
                "子问题5: 问题的利益相关方或涉及主体是谁？",
                "子问题6: 需要比较或对标的参照物是什么？（如适用）",
                "子问题7: 问题的量化指标或评估标准是什么？",
                "子问题8: 是否存在需要排除的特殊情况或边界条件？"
              ]
            }}

            用户主要问题："{initial_query}"
        """,
        "reflection_prompt_template": """
            作为资深的分析评估员，请评估为回答以下用户原始问题而收集的信息摘要。

            用户原始问题："{initial_query}"

            目前收集到的信息摘要（可能部分截断）：
            {memory_context_for_llm}

            请评估：
            1.  `can_answer`: 这些信息是否**足够全面**地回答用户的原始问题？(true/false)
            2.  `irrelevant_urls`: 当前摘要中，是否有与回答原始问题**明显无关或信息价值低**的条目？（仅列出这些条目的来源 URL 列表，如果没有则为空列表 []）
            3.  `new_subqueries`: 基于当前信息和原始问题，还需要提出哪些**具体的、新的**子问题来**获取关键信息数据**或**澄清当前状况**？（例如：需要特定指标的最新数据？需要更详细的信息？需要某事件的最新进展？如果信息已足够，则返回空列表 []）

            请严格按照以下 JSON 格式进行响应，不要包含任何额外的解释或评论：
            {{
                "can_answer": boolean,
                "irrelevant_urls": ["url1", "url2", ...],
                "new_subqueries": ["新问题1", "新问题2", ...]
            }}
        """,
        "synthesis_prompt_template": """
            您是一位资深的分析师。您的任务是基于以下通过网络搜索收集到的信息片段，为用户生成一份全面、结构清晰、客观中立的信息研究报告，以回答他们的原始问题。

            用户的原始问题是："{initial_query}"

            以下是收集到的相关信息（主要来自新闻、网页摘要）：
            --- 开始信息 ---
            {final_memory_context}
            --- 结束信息 ---
            {analysis_section}
            请严格遵守以下要求撰写报告：
            1.  **完全基于**上面提供的信息片段撰写报告。不得添加任何外部知识、个人观点、未经证实的精确数据（除非信息片段中明确提到）。
            2.  清晰、有条理地组织报告内容，直接回答用户的原始问题。可适当使用标题和小标题。
            3.  在报告中**必须**引用信息来源。当您使用某条信息时，请在其后用方括号注明来源 URL，格式为 `[来源: URL]`。例如：XX 公司发布了新产品 [来源: http://example.com/news1]。**确保URL完整且在方括号内**。
            4.  如果提供了"数据扫描摘要"，请将扫描结果（如关键词频率、发现的数值或百分比）适当地融入报告内容中，并指明这只是基于所提供文本的初步扫描。**不要将扫描到的数字当作精确的实时数据**。
            5.  语言专业、客观、中立。避免使用过度乐观或悲观的词语。专注于总结和呈现收集到的信息。
            6.  如果信息片段之间存在矛盾或不一致之处，请客观地指出。
            7.  如果收集到的信息不足以回答问题的某些方面，请在报告中明确说明。
            8.  报告结尾可以根据信息做一个简要的总结或展望，但必须基于已提供的信息，并保持客观。

            请开始撰写您的分析报告：
        """
    },
    "finance": {
        "name": "金融市场 (Financial Markets)",
        "filename_prefix": "金融市场分析_",
        "llm_system_prompt_assistant": "你是一位专门研究金融市场行情的资深分析助理。请仔细遵循指示并以要求的格式回答。",
        "llm_system_prompt_synthesizer": "你是一位专业的金融市场分析师，负责根据收集的信息生成客观的市场分析报告。请严格遵循指示，并使用 `[来源: URL]` 格式引用来源。",
        "analyzer_keywords": [
            '股票', 'A股', '港股', '美股', '上证指数', '深证成指', '创业板指', '恒生指数', '纳斯达克', '道琼斯',
            '涨', '跌', '上涨', '下跌', '涨幅', '跌幅', '成交额', '成交量', '换手率',
            '市盈率', '市净率', '概念', '板块', '行业', '龙头',
            '宏观经济', '利率', '通胀', '加息', '降息', '财报', '业绩',
            '利好', '利空', '风险', '机会', '预期', '预测', '分析', '行情', 'IPO', '并购', '央行'
        ],
        "plan_prompt_template": """
            请将以下用户的关于【{industry_name}】的主要问题分解为具体的、可搜索的子问题列表，以全面分析该主题。
            请关注市场概览、关键指标/指数表现、重要板块/公司动态、相关新闻事件、宏观经济/政策影响、技术趋势（如果信息允许）等方面。
            请严格按照以下 JSON 格式输出，不要包含任何额外的解释或评论：
            {{
              "subqueries": [
                "子问题1: {industry_name}整体表现如何？",
                "子问题2: 主要基准（如指数、利率）情况怎样？",
                "子问题3: 有哪些值得关注的热点领域或概念？",
                "子问题4: 有哪些重要的行业新闻或政策发布？",
                "子问题5: （可选，如适用）特定公司或资产的表现和相关信息？"
              ]
            }}

            用户主要问题："{initial_query}"
        """,
        "reflection_prompt_template": """
            作为【{industry_name}】分析评估员，请评估为回答以下用户原始问题而收集的信息摘要。

            用户原始问题："{initial_query}"

            目前收集到的信息摘要（可能部分截断）：
            {memory_context_for_llm}

            请评估：
            1.  `can_answer`: 这些信息是否**足够全面**地回答用户的原始【{industry_name}】问题？(true/false)
            2.  `irrelevant_urls`: 当前摘要中，是否有与回答原始问题**明显无关或信息价值低**的条目？（仅列出这些条目的来源 URL 列表，如果没有则为空列表 []）
            3.  `new_subqueries`: 基于当前信息和原始问题，还需要提出哪些**具体的、新的**子问题来**获取关键行业数据**或**澄清当前状况**？（例如：需要特定指标的最新数据？需要某领域更详细的动态？需要某事件的最新进展？如果信息已足够，则返回空列表 []）

            请严格按照以下 JSON 格式进行响应，不要包含任何额外的解释或评论：
            {{
                "can_answer": boolean,
                "irrelevant_urls": ["url1", "url2", ...],
                "new_subqueries": ["新问题1", "新问题2", ...]
            }}
        """,
        "synthesis_prompt_template": """
            您是一位专业的【{industry_name}】分析师。您的任务是基于以下通过网络搜索收集到的信息片段，为用户生成一份全面、结构清晰、客观中立的行业分析报告，以回答他们的原始问题。

            用户的原始问题是："{initial_query}"

            以下是收集到的相关行业信息（主要来自新闻、网页摘要）：
            --- 开始信息 ---
            {final_memory_context}
            --- 结束信息 ---
            {analysis_section}
            请严格遵守以下要求撰写报告：
            1.  **完全基于**上面提供的信息片段撰写报告。不得添加任何外部知识、个人观点、未经证实的精确数据（除非信息片段中明确提到）。
            2.  清晰、有条理地组织报告内容，直接回答用户的原始问题。可适当使用标题和小标题（例如：行业概览、主要趋势、关键参与者、重要新闻、总结与展望等）。
            3.  在报告中**必须**引用信息来源。当您使用某条信息时，请在其后用方括号注明来源 URL，格式为 `[来源: URL]`。例如：XX 公司发布了新产品 [来源: http://example.com/news1]。**确保URL完整且在方括号内**。
            4.  如果提供了"数据扫描摘要"，请将扫描结果（如关键词频率、发现的数值或百分比）适当地融入报告内容中，并指明这只是基于所提供文本的初步扫描。**不要将扫描到的数字当作精确的实时数据**。
            5.  语言专业、客观、中立。避免使用过度乐观或悲观的词语，避免给出直接的商业建议。专注于总结和呈现收集到的信息。
            6.  如果信息片段之间存在矛盾或不一致之处，请客观地指出。
            7.  如果收集到的信息不足以回答问题的某些方面，请在报告中明确说明。
            8.  报告结尾可以根据信息做一个简要的总结或展望，但必须基于已提供的信息，并保持客观。

            请开始撰写您的【{industry_name}】分析报告：
        """
    },
    "tech": {
        "name": "科技行业 (Technology Industry)",
        "filename_prefix": "科技行业分析_",
        "llm_system_prompt_assistant": "你是一位专门研究科技行业动态的资深分析助理。请仔细遵循指示并以要求的格式回答。",
        "llm_system_prompt_synthesizer": "你是一位专业的科技行业分析师，负责根据收集的信息生成客观的行业分析报告。请严格遵循指示，并使用 `[来源: URL]` 格式引用来源。",
        "analyzer_keywords": [
            '人工智能', 'AI', '机器学习', '芯片', '半导体', '云计算', '大数据', '软件', '硬件',
            '互联网', 'SaaS', 'PaaS', 'IaaS', '物联网', 'IoT', '5G', '6G', '元宇宙', 'VR', 'AR',
            '初创公司', '融资', '风险投资', 'VC', 'PE', '上市', 'IPO', '裁员', '并购', 'M&A',
            '科技巨头', '苹果', '谷歌', '微软', '亚马逊', 'Meta', '腾讯', '阿里巴巴', '华为', '字节跳动',
            '创新', '研发', '专利', '趋势', '法规', '监管', '数据隐私', '网络安全'
        ],
        "plan_prompt_template": """
            请将以下用户的关于【{industry_name}】的主要问题分解为具体的、可搜索的子问题列表，以全面分析该主题。
            请关注市场规模与增长、关键技术领域、主要公司动态（产品、战略、财报）、投融资活动、最新行业新闻、政策法规影响等方面。
            请严格按照以下 JSON 格式输出，不要包含任何额外的解释或评论：
            {{
              "subqueries": [
                "子问题1: {industry_name}整体发展趋势如何？",
                "子问题2: 主要技术领域（如AI、云计算）有哪些新进展？",
                "子问题3: 重点科技公司最近有哪些重要动态？",
                "子问题4: {industry_name}最近的投融资情况怎样？",
                "子问题5: 有哪些值得关注的行业新闻或政策发布？"
              ]
            }}

            用户主要问题："{initial_query}"
        """,
        "reflection_prompt_template": """
            作为【{industry_name}】分析评估员，请评估为回答以下用户原始问题而收集的信息摘要。

            用户原始问题："{initial_query}"

            目前收集到的信息摘要（可能部分截断）：
            {memory_context_for_llm}

            请评估：
            1.  `can_answer`: 这些信息是否**足够全面**地回答用户的原始【{industry_name}】问题？(true/false)
            2.  `irrelevant_urls`: 当前摘要中，是否有与回答原始问题**明显无关或信息价值低**的条目？（仅列出这些条目的来源 URL 列表，如果没有则为空列表 []）
            3.  `new_subqueries`: 基于当前信息和原始问题，还需要提出哪些**具体的、新的**子问题来**获取关键行业数据**或**澄清当前状况**？（例如：需要某项技术的最新应用？需要某公司更详细的战略分析？需要某事件的最新进展？如果信息已足够，则返回空列表 []）

            请严格按照以下 JSON 格式进行响应，不要包含任何额外的解释或评论：
            {{
                "can_answer": boolean,
                "irrelevant_urls": ["url1", "url2", ...],
                "new_subqueries": ["新问题1", "新问题2", ...]
            }}
        """,
        "synthesis_prompt_template": """
            您是一位专业的【{industry_name}】分析师。您的任务是基于以下通过网络搜索收集到的信息片段，为用户生成一份全面、结构清晰、客观中立的行业分析报告，以回答他们的原始问题。

            用户的原始问题是："{initial_query}"

            以下是收集到的相关行业信息（主要来自新闻、网页摘要）：
            --- 开始信息 ---
            {final_memory_context}
            --- 结束信息 ---
            {analysis_section}
            请严格遵守以下要求撰写报告：
            1.  **完全基于**上面提供的信息片段撰写报告。不得添加任何外部知识、个人观点、未经证实的精确数据（除非信息片段中明确提到）。
            2.  清晰、有条理地组织报告内容，直接回答用户的原始问题。可适当使用标题和小标题（例如：行业概览、技术趋势、主要公司动态、投融资情况、总结与展望等）。
            3.  在报告中**必须**引用信息来源。当您使用某条信息时，请在其后用方括号注明来源 URL，格式为 `[来源: URL]`。例如：XX 公司发布了新AI模型 [来源: http://example.com/news1]。**确保URL完整且在方括号内**。
            4.  如果提供了"数据扫描摘要"，请将扫描结果（如关键词频率、发现的数值或百分比）适当地融入报告内容中，并指明这只是基于所提供文本的初步扫描。**不要将扫描到的数字当作精确的实时数据**。
            5.  语言专业、客观、中立。避免使用过度乐观或悲观的词语，避免给出直接的商业建议。专注于总结和呈现收集到的信息。
            6.  如果信息片段之间存在矛盾或不一致之处，请客观地指出。
            7.  如果收集到的信息不足以回答问题的某些方面，请在报告中明确说明。
            8.  报告结尾可以根据信息做一个简要的总结或展望，但必须基于已提供的信息，并保持客观。

            请开始撰写您的【{industry_name}】分析报告：
        """
    },
}

# --- 核心函数 ---

def websearch(query, count=5):
    """执行网络搜索"""
    logging.info(f"正在执行网络搜索: '{query}'")
    url = "https://api.bochaai.com/v1/web-search"
    payload = json.dumps({
        "query": query,
        "summary": True,
        "count": count,
        "page": 1
    })
    headers = {
        'Authorization': SEARCH_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        # logging.info(f'搜索到的新闻内容: {data}')
    except requests.exceptions.Timeout:
        logging.error(f"查询 '{query}' 的网络搜索请求超时")
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"查询 '{query}' 的网络搜索请求期间出错: {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"解码网络搜索 '{query}' 的 JSON 时出错: {e}。响应文本: {response.text[:500]}...")
        return []
    except Exception as e:
        logging.error(f"查询 '{query}' 的网络搜索期间发生意外错误: {e}", exc_info=True)
        return []

    webpages_data = data.get('data', {}).get('webPages', {})
    value_list = webpages_data.get('value')

    if value_list is None or not isinstance(value_list, list):
        logging.warning(f"在查询 '{query}' 的响应中找不到 'value' 列表或它不是一个列表。")
        return []

    filtered_results = [
        item for item in value_list
        if item.get('url') and (item.get('summary') or item.get('snippet'))
    ]
    logging.info(f"查询 '{query}' 的网络搜索返回了 {len(filtered_results)} 个有效结果。")
    return filtered_results

def qwen_llm(prompt, industry_config, model="qwen-max", response_format=None):
    """调用 Qwen LLM"""
    logging.info(f"正在调用 Qwen LLM (模型: {model}) 处理: {prompt[:100]}...")
    system_message_content = industry_config.get("llm_system_prompt_assistant", "你是一个有用的助手。")

    try:
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

        completion_args = {
            "model": model,
            "messages": [
                {'role': 'system', 'content': system_message_content},
                {'role': 'user', 'content': prompt}
            ],
            "temperature": 0.2,
        }
        if response_format:
            completion_args["response_format"] = response_format
            logging.info("正在请求 LLM 返回 JSON 格式。")

        completion = client.chat.completions.create(**completion_args)
        content = completion.choices[0].message.content
        logging.info("LLM 调用成功。")
        return content
    except Exception as e:
        logging.error(f"调用 Qwen LLM 时出错: {e}", exc_info=True)
        return None

def deepseek_stream(prompt, industry_config, model_name="deepseek-r1"):
    """调用 Deepseek LLM 流式生成报告"""
    logging.info(f"正在调用 Deepseek LLM 流 (模型: {model_name}) 进行最终合成...")
    system_message_content = industry_config.get(
        "llm_system_prompt_synthesizer",
        "你是一个有用的助手，负责将信息合成为最终报告，并使用 [来源: URL] 格式仔细引用来源。"
    )

    try:
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_message_content},
                {'role': 'user', 'content': prompt}
            ],
            stream=True,
            temperature=0.5,
        )

        def generate():
            for chunk in stream:
                if not getattr(chunk, 'choices', None) and hasattr(chunk, 'usage') and chunk.usage:
                    continue
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content_piece = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'content': content_piece})}\n\n"
            yield "data: [DONE]\n\n"

        return generate()
    except Exception as e:
        logging.error(f"调用 Deepseek LLM 流时出错: {e}", exc_info=True)
        def error_generate():
            yield f"data: {json.dumps({'error': '生成报告时出错'})}\n\n"
            yield "data: [DONE]\n\n"
        return error_generate()

def simple_data_analyzer(text_data, industry_config):
    """执行基本数据分析"""
    industry_name = industry_config.get("name", "数据")
    numbers = []
    percentages = []
    keywords = Counter()

    number_pattern = re.compile(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?:%|亿|万|千|百)?')
    percentage_pattern = re.compile(r'([+-]?\d+(?:\.\d+)?)\s*%')

    relevant_keywords = industry_config.get("analyzer_keywords", [])
    if not relevant_keywords:
        logging.warning("在行业配置中未找到 analyzer_keywords。将跳过关键词分析。")

    full_text = " ".join(text_data)
    logging.info(f"正在对 {len(text_data)} 个文本片段（{industry_name} 上下文）执行基本分析。")

    num_values = number_pattern.findall(full_text)
    numbers.extend([float(val.replace(',', '')) for val in num_values if val and val.replace(',', '').replace('.', '', 1).isdigit()])
    logging.info(f"找到 {len(numbers)} 个潜在的数值。")

    perc_values = percentage_pattern.findall(full_text)
    percentages.extend([float(p) for p in perc_values])
    logging.info(f"找到 {len(percentages)} 个百分比值。")

    if relevant_keywords:
        lower_full_text = full_text.lower()
        for keyword in relevant_keywords:
            count = lower_full_text.count(keyword.lower())
            if count > 0:
                keywords[keyword] += count
        logging.info(f"关键词计数 (Top 5): {keywords.most_common(5)}")
    else:
         logging.info("由于未提供关键词，跳过关键词分析。")

    analysis_summary = "简要数据扫描摘要:\n"
    found_data = False

    if numbers:
        found_data = True
        try:
            analysis_summary += f"- 扫描到 {len(numbers)} 个数值点. 平均值: {sum(numbers)/len(numbers):.2f}, 最小值: {min(numbers):.2f}, 最大值: {max(numbers):.2f}\n"
        except ZeroDivisionError:
             analysis_summary += f"- 扫描到 {len(numbers)} 个数值点, 但无法计算统计数据。\n"
    else:
        analysis_summary += "- 未明确扫描到可用于统计分析的数值点。\n"

    if percentages:
        found_data = True
        try:
            analysis_summary += f"- 扫描到 {len(percentages)} 个百分比值. 平均值: {sum(percentages)/len(percentages):.2f}%, 最小值: {min(percentages):.2f}%, 最大值: {max(percentages):.2f}%\n"
        except ZeroDivisionError:
             analysis_summary += f"- 扫描到 {len(percentages)} 个百分比值, 但无法计算统计数据。\n"
    else:
        analysis_summary += "- 未扫描到明确的百分比值。\n"

    if keywords:
        found_data = True
        analysis_summary += f"- 主要相关关键词频率: " + ", ".join([f"{k}({v})" for k, v in keywords.most_common(5)]) + "\n"
    elif relevant_keywords:
        analysis_summary += "- 未扫描到相关的关键词。\n"

    return analysis_summary if found_data else "未在收集的信息中发现足够的可量化数据进行扫描分析。"

def deep_research_workflow(initial_query, industry_config, max_iterations=3):
    """协调整个行业分析过程"""
    industry_name = industry_config.get("name", "Selected Industry")
    logging.info(f"开始为查询 '{initial_query}' 进行 {industry_name} 分析")

    memory = []
    processed_urls = set()
    current_subqueries = []
    all_subqueries_history = set()

    plan_template = industry_config.get("plan_prompt_template")
    reflection_template = industry_config.get("reflection_prompt_template")
    synthesis_template = industry_config.get("synthesis_prompt_template")


    if not all([plan_template, reflection_template, synthesis_template]):
        logging.error(f"行业 '{industry_name}' 的配置缺少一个或多个必需的提示模板。")
        return None

    for iteration in range(max_iterations):
        logging.info(f"--- 开始 {industry_name} 分析迭代 {iteration + 1} ---")

        if iteration == 0:
            plan_prompt = plan_template.format(
                industry_name=industry_name,
                initial_query=initial_query
            )
            logging.info(f"正在生成初始 {industry_name} 子查询...")
            llm_response = qwen_llm(
                plan_prompt,
                industry_config=industry_config,
                response_format={"type": "json_object"}
            )

            if not llm_response:
                logging.error("从 LLM 获取初始规划响应失败。")
                current_subqueries = [initial_query]
                logging.warning(f"回退到使用初始查询: {initial_query}")
            else:
                try:
                    plan_result = json.loads(llm_response)
                    current_subqueries = plan_result.get('subqueries', [])
                    if not current_subqueries or not isinstance(current_subqueries, list):
                         logging.warning(f"LLM 返回了 JSON，但 'subqueries' 键丢失、无效或为空。响应: {llm_response}")
                         current_subqueries = [initial_query]
                         logging.warning(f"回退到使用初始查询: {initial_query}")
                    else:
                        current_subqueries = [q for q in current_subqueries if isinstance(q, str) and q.strip()]
                        logging.info(f"生成的初始子查询: {current_subqueries}")
                except json.JSONDecodeError as e:
                    logging.error(f"解码初始规划的 JSON 时失败: {e}。响应: {llm_response[:500]}...")
                    current_subqueries = [initial_query]
                    logging.warning(f"回退到使用初始查询: {initial_query}")
                except Exception as e:
                    logging.error(f"解析初始规划时发生意外错误: {e}", exc_info=True)
                    current_subqueries = [initial_query]
                    logging.warning(f"回退到使用初始查询: {initial_query}")

        elif not current_subqueries:
             logging.info("上一步反思未生成新的子查询。结束迭代周期。")
             break

        subqueries_to_search = [q for q in current_subqueries if q and q not in all_subqueries_history]
        logging.info(f"本次迭代选择的子查询: {subqueries_to_search}")

        if not subqueries_to_search and iteration > 0:
             logging.info("所有生成的子查询都已被搜索或列表为空。转到反思阶段。")

        new_results_count = 0
        for subquery in subqueries_to_search:
            if not subquery: continue
            logging.info(f"正在为 {industry_name} 信息搜索网络: '{subquery}'...")
            all_subqueries_history.add(subquery)
            search_results = websearch(subquery)
            time.sleep(1.5)

            for result in search_results:
                url = result.get('url')
                if url and url not in processed_urls:
                    processed_urls.add(url)
                    summary = result.get('summary', '') or result.get('snippet', '')
                    if summary:
                        memory.append({
                            "subquery": subquery,
                            "url": url,
                            "name": result.get('name', 'N/A'),
                            "summary": summary,
                            "snippet": result.get('snippet', '')
                        })
                        new_results_count += 1
                    else:
                        logging.debug(f"跳过没有摘要/片段的结果: {url}")

        logging.info(f"添加了 {new_results_count} 个新的独立结果。总内存大小: {len(memory)}")

        memory_context_for_llm = ""
        if memory:
             context_items = []
             token_estimate = 0
             max_tokens_estimate = 10000
             for item in reversed(memory):
                 item_text = f"  - 查询 '{item['subquery']}': {item['summary'][:250]}... (来源: {item['url']})\n"
                 token_estimate += len(item_text) / 2
                 if token_estimate > max_tokens_estimate:
                     logging.warning("用于反思的内存上下文因估计的 token 限制而被截断。")
                     break
                 context_items.append(item_text)
             memory_context_for_llm = "".join(reversed(context_items))
        else:
            memory_context_for_llm = "当前没有收集到任何相关信息。"

        reflection_prompt = reflection_template.format(
            industry_name=industry_name,
            initial_query=initial_query,
            memory_context_for_llm=memory_context_for_llm
        )
        logging.info(f"正在反思收集到的 {industry_name} 数据...")
        llm_response = qwen_llm(
            reflection_prompt,
            industry_config=industry_config,
            response_format={"type": "json_object"}
            )

        can_answer = False
        current_subqueries = []

        if not llm_response:
            logging.error("从 LLM 获取反思响应失败。")
        else:
            try:
                reflection_result = json.loads(llm_response)

                can_answer = reflection_result.get('can_answer', False)
                irrelevant_urls_list = reflection_result.get('irrelevant_urls', [])
                new_subqueries_list = reflection_result.get('new_subqueries', [])

                if not isinstance(can_answer, bool):
                    logging.warning(f"LLM 为 'can_answer' 返回了无效类型。默认为 False。值: {can_answer}")
                    can_answer = False

                if not isinstance(irrelevant_urls_list, list):
                     logging.warning(f"LLM 为 'irrelevant_urls' 返回了无效类型。默认为空列表。值: {irrelevant_urls_list}")
                     irrelevant_urls = set()
                else:
                     irrelevant_urls = set(u for u in irrelevant_urls_list if isinstance(u, str))

                if not isinstance(new_subqueries_list, list):
                    logging.warning(f"LLM 为 'new_subqueries' 返回了无效类型。默认为空列表。值: {new_subqueries_list}")
                    current_subqueries = []
                else:
                     current_subqueries = [q for q in new_subqueries_list if isinstance(q, str) and q.strip()]

                logging.info(f"反思 - 能否回答 {industry_name} 查询: {can_answer}")

                if irrelevant_urls:
                    logging.info(f"反思 - 发现 {len(irrelevant_urls)} 个可能不相关的项目需要修剪。")
                    original_memory_size = len(memory)
                    memory = [item for item in memory if item['url'] not in irrelevant_urls]
                    logging.info(f"内存从 {original_memory_size} 项修剪到 {len(memory)} 项。")

                if can_answer:
                    logging.info(f"反思完成: {industry_name} 信息被认为足够。")
                    break

                if not current_subqueries and iteration < max_iterations - 1:
                    logging.warning("反思：信息不足，但未建议新的有效子查询。继续可能导致没有进展的循环。")
                elif current_subqueries:
                     logging.info(f"反思 - 需要新的 {industry_name} 子查询: {current_subqueries}")

            except json.JSONDecodeError as e:
                logging.error(f"解码反思的 JSON 时失败: {e}。响应: {llm_response[:500]}...")
            except Exception as e:
                logging.error(f"处理反思 JSON 时发生意外错误: {e}", exc_info=True)

        if iteration == max_iterations - 1:
            logging.warning("达到最大迭代次数。")
            if not can_answer:
                 logging.warning(f"继续进行合成，尽管 {industry_name} 信息可能不完整。")

    if not memory:
        logging.error(f"在 {max_iterations} 次迭代后未收集到 {industry_name} 信息。无法生成分析报告。")
        return None

    needs_analysis = len(memory) >= 4
    analysis_summary = ""

    if needs_analysis:
        logging.info("发现足够的数据，尝试进行基本数据扫描。")
        texts_for_analysis = [item['summary'] for item in memory]
        analysis_summary = simple_data_analyzer(texts_for_analysis, industry_config)
    else:
        logging.info("数据不足以进行有意义的扫描，跳过数据分析。")

    logging.info(f"正在准备最终 {industry_name} 报告的上下文...")
    final_memory_context = "\n\n".join([
        f"来源 URL: {item['url']}\n相关子问题: {item['subquery']}\n标题/名称: {item['name']}\n内容摘要: {item['summary']}"
        for item in memory
    ])

    analysis_section = ""
    if analysis_summary and "未在收集的信息中发现足够的可量化数据进行扫描分析" not in analysis_summary:
        analysis_section = f"\n\n补充数据扫描摘要:\n{analysis_summary}\n"

    synthesis_prompt = synthesis_template.format(
        industry_name=industry_name,
        initial_query=initial_query,
        final_memory_context=final_memory_context,
        analysis_section=analysis_section
    )

    return synthesis_prompt, industry_config

# --- FastAPI 路由 ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """渲染首页"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "industries": INDUSTRY_CONFIGS.keys(),
        "industry_names": {k: v["name"] for k, v in INDUSTRY_CONFIGS.items()}
    })

@app.post("/generate-report")
async def generate_report(
    query: str = Form(...),
    industry: str = Form(...),
    max_iterations: int = Form(3)
):
    """生成报告流式响应"""
    industry_config = INDUSTRY_CONFIGS.get(industry)
    if not industry_config:
        return {"error": "无效的行业选择"}
    
    synthesis_prompt, industry_config = deep_research_workflow(
        query,
        industry_config=industry_config,
        max_iterations=max_iterations
    )
    
    if not synthesis_prompt:
        return {"error": "无法生成报告提示"}
    
    return StreamingResponse(
        deepseek_stream(synthesis_prompt, industry_config),
        media_type="text/event-stream"
    )

@app.post("/save-report")
async def save_report(
    query: str = Form(...),
    industry: str = Form(...),
    report_content: str = Form(...)
):
    """保存生成的报告"""
    industry_config = INDUSTRY_CONFIGS.get(industry)
    if not industry_config:
        return {"error": "无效的行业选择"}
    
    # 为报告创建唯一文件名
    safe_query_part = re.sub(r'[^\w\s-]', '', query[:30]).strip()
    safe_query_part = re.sub(r'[-\s]+', '_', safe_query_part)
    filename = f"{industry_config.get('filename_prefix', '分析报告_')}{safe_query_part}_{uuid.uuid4().hex[:8]}.txt"
    
    # 保存报告
    filepath = REPORTS_DIR / filename
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"分析主题：{industry_config['name']}\n")
            f.write(f"分析问题：{query}\n\n")
            f.write("="*10 + " 分析报告 " + "="*10 + "\n\n")
            f.write(report_content)
        return {"success": True, "filename": filename}
    except Exception as e:
        logging.error(f"保存报告时出错: {e}")
        return {"error": "保存报告失败"}


# --- 主程序 ---
if __name__ == "__main__":
    import uvicorn
    # 启动 FastAPI 应用
    uvicorn.run(app, host="0.0.0.0", port=8000)