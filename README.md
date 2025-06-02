# Deep Research demo - README

## 功能概述

本功能复现Deep Research，结合网络搜索和大型语言模型(LLM)技术，快速生成专业、结构化的行业分析报告。主要功能包括：

1. **智能问题分解** - 将用户输入的行业问题自动分解为多个可搜索的子问题
2. **网络搜索集成** - 通过Bochaai API获取最新的网络信息
3. **信息评估与迭代** - 自动评估收集的信息质量，必要时进行补充搜索
4. **数据分析** - 对收集的文本进行基本数据分析（关键词频率、数值提取等）
5. **报告生成** - 使用Deepseek LLM生成结构清晰、引用准确的行业报告
6. **多行业支持** - 目前支持金融、科技等多个行业的专业分析


deep Research（使用网络搜索和LLM） - 带FastAPI前端

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

## 架构图
![架构图](/doc/deepResearch.png)(https://github.com/meetshawn/DeepResearch-lite/blob/main/doc/deepResearch.png)


## 部署指南

### 环境要求

- Python 3.x+
- 有效的API密钥：
  - Bochaai API密钥（用于网络搜索）
  - 阿里Dashscope API密钥（用于LLM调用）

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/meetshawn/DeepResearch-lite.git
   cd DeepResearch-lite
   ```

2. 创建并激活虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

4. 配置环境变量：
   在项目根目录创建`.env`文件，内容如下：
   ```
   BOCHAAI_API_KEY=your_bochaai_api_key
   DASHSCOPE_API_KEY=your_dashscope_api_key
   DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1
   ```

### 运行应用

1. 启动FastAPI服务：
   ```bash
   python main.py
   ```
   或使用uvicorn直接运行：
   ```bash
   uvicorn main:app --reload
   ```

2. 访问Web界面：
   打开浏览器访问 `http://localhost:8000`

## 使用说明

1. **输入查询**：
   - 在首页输入框中输入您想分析的行业问题（如"当前AI芯片市场发展趋势"）
   - 选择行业类型（金融、科技等）

2. **生成报告**：
   - 点击"生成报告"按钮
   - 系统将开始自动搜索、分析和生成报告
   - 报告内容将实时显示在页面上

3. **保存报告**：
   - 报告生成完成后，可以点击"保存报告"按钮
   - 报告将以文本文件形式保存在`reports/`目录下

## 配置选项

您可以通过修改`INDUSTRY_CONFIGS`字典来调整不同行业的分析参数：

- `llm_system_prompt_assistant`: 用于规划和分析的LLM系统提示
- `llm_system_prompt_synthesizer`: 用于报告生成的LLM系统提示
- `analyzer_keywords`: 行业特定关键词用于数据分析
- 各种提示模板（plan/reflection/synthesis）

## 注意事项

1. API调用限制：
   - 请确保您的API密钥有足够的配额
   - 系统会自动控制请求频率以避免超限

2. 报告质量：
   - 报告内容完全基于网络搜索结果
   - 建议验证报告中引用的关键数据和事实

3. 性能考虑：
   - 复杂查询可能需要较长时间（2-5分钟）
   - 可以通过调整`max_iterations`参数控制搜索深度

## 贡献指南

欢迎贡献代码或提出改进建议！请通过issue或pull request提交您的想法。

## 许可证

本项目采用MIT许可证。详情请参阅LICENSE文件。