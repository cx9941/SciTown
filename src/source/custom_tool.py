from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_community.utilities import SearxSearchWrapper
from crewai.tools.agent_tools.base_agent_tools import BaseAgentTool
from crewai.agents.agent_builder.base_agent import BaseAgent#######
import logging
from typing import List###############

class WebSearchToolSchema(BaseModel):
    """Input schema for WebSearchTool"""
    query: str = Field(..., description="搜索关键词或问题")
    context: str = Field("", description="补充上下文信息（可选）")
    coworker: Optional[str] = Field(None, description="可指定协作者名称（若需协作分析结果）")
    max_results: Optional[int] = Field(3, description="返回的最大结果数（默认3）")
    time_range: Optional[str] = Field(None, description="时间范围过滤，格式如'2023-01-01'")

class WebSearchTool(BaseAgentTool):
    """Tool for performing online searches using SearxNG"""
    name: str = "web_search"
    description: str = "使用SearxNG搜索引擎从互联网获取最新信息，支持时间范围过滤和结果数量限制。"
    args_schema: Type[BaseModel] = WebSearchToolSchema
    agents: List[BaseAgent] = []  # 新增：显式定义必填字段，默认空列表（根据需求调整）

    def _run(
        self,
        query: str,
        context: str = "",
        coworker: Optional[str] = None,
        max_results: int = 3,##默认返回3个，可以改多点
        time_range: Optional[str] = None,
        **kwargs
    ) -> str:
        try:
            # 1. 增强查询（融入上下文）
            enhanced_query = f"{query} {context}".strip() if context else query
            
            # 2. 使用您原有的搜索函数
            results = self.search_online(
                query=enhanced_query,
                time_range=time_range
            )
            
            # 3. 格式化结果
            return self._format_results(results, max_results)
            
        except Exception as e:
            logging.error(f"搜索失败: {str(e)}")
            return f"搜索失败: {str(e)}"

    def search_online(self, query: str, time_range: str = None):
        Searx = SearxSearchWrapper(
            searx_host="https://searxng.cstcloud.cn"  # 硬编码地址
        )
        if time_range is None:
            time_range = time_range  # 这里才引用 self.time_range
        if time_range:
            query += f" before:{time_range}" 
        search_params = {
            "query": query,
            "num_results": 40,
            "language": "en",  
        }
        results = Searx.results(**search_params)
        #print(results)
        return results

    def _format_results(self, results: list, max_results: int) -> str:
        """格式化搜索结果（Markdown）"""
        if not results:
            return "未找到相关结果。"
        #print("--------------------------------------格式化搜索-------------------------------------------------")
        formatted = []
        for idx, item in enumerate(results[:max_results], 1):
            title = item.get('title', '无标题')
            link = item.get('link', '#')  # 注意：样例中显示的是'link'而不是'url'
            snippet = item.get('snippet', '无描述')
            formatted.append(
                f"{idx}. [{title}]({link})\n   {snippet}"
            )
        return "\n\n".join(formatted)
