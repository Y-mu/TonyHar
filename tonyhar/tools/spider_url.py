"""数学计算工具。"""

from typing import Any

from tonyhar.tooling import BaseTool, ToolPolicy, tool

import requests
import asyncio


@tool(policy=ToolPolicy(parallel_safe=True))
class GetUrlTool(BaseTool):
    """爬取url地址下的内容、md返回"""

    name = "spider_url"
    description = "输入url地址、爬取内容后返回"
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "需要抓取的 URL。",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    async def execute(self, url: str) -> str:
        response = await asyncio.to_thread(
            requests.get,
            url,
            timeout=15,
        )
        status = response.raise_for_status()
        code = response.status_code
        # breakpoint()
        return response.text


if __name__ == "__main__":
    async def main():
        tool = GetUrlTool()

        content = await tool.execute(
            url="https://blog.csdn.net/m0_46279060/article/details/159397624"
        )

        print(content[:500])

        
    asyncio.run(main())
    