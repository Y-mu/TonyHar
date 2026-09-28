"""网页正文抓取与 Markdown 清洗工具。"""

import asyncio
import html
import re
import unicodedata

import requests
import trafilatura

from tonyhar.tooling import BaseTool, ToolPolicy, tool


_HORIZONTAL_WHITESPACE = re.compile(
    r"[\t\f\v \u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+"
)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_MARKDOWN_BLOCK = re.compile(
    r"^(?:"
    r"\s{4,}\S|"
    r"\s{0,3}#{1,6}\s+|"
    r"\s{0,3}>|"
    r"\s{0,3}(?:[-+*]|\d+[.)])\s+|"
    r"\s{0,3}(?:-{3,}|_{3,}|\*{3,})\s*$|"
    r"\s{0,3}\[[^]]+\]:\s*|"
    r"\s*</?[A-Za-z][^>]*>\s*$"
    r")"
)
_INVISIBLE_CHARACTERS = str.maketrans("", "", "\ufeff\u200b\u2060\u00ad")
_FULLWIDTH_ALPHANUMERIC = str.maketrans(
    {
        **{chr(code + 0xFEE0): chr(code) for code in range(ord("0"), ord("9") + 1)},
        **{chr(code + 0xFEE0): chr(code) for code in range(ord("A"), ord("Z") + 1)},
        **{chr(code + 0xFEE0): chr(code) for code in range(ord("a"), ord("z") + 1)},
    }
)
_CJK_RANGES = (
    ("\u3400", "\u4dbf"),
    ("\u4e00", "\u9fff"),
    ("\uf900", "\ufaff"),
)
_NO_SPACE_AFTER = set("([{《〈「『【〔〖〘〚")
_NO_SPACE_BEFORE = set(")]},.!?;:》〉」』】〕〗〙〛，。！？；：、")


def _is_cjk(character: str) -> bool:
    return any(start <= character <= end for start, end in _CJK_RANGES)


def _join_prose_lines(lines: list[str]) -> str:
    """合并网页排版产生的软换行，同时避免给中文句子插入空格。"""
    joined = lines[0]
    for line in lines[1:]:
        left = joined[-1]
        right = line[0]
        needs_space = not (
            _is_cjk(left)
            or _is_cjk(right)
            or left in _NO_SPACE_AFTER
            or right in _NO_SPACE_BEFORE
        )
        joined += (" " if needs_space else "") + line
    return joined


def clean_extracted_markdown(text: str) -> str:
    """规范抓取到的 Markdown，保留其块级结构以便后续切块。

    该函数只处理可确定的格式噪声，不尝试自动修正文中的错别字、术语或事实。
    """
    if not text:
        return ""

    normalized = html.unescape(text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u2028", "\n").replace("\u2029", "\n")
    normalized = unicodedata.normalize("NFC", normalized)
    normalized = normalized.translate(_FULLWIDTH_ALPHANUMERIC)
    normalized = normalized.translate(_INVISIBLE_CHARACTERS)
    normalized = _CONTROL_CHARACTERS.sub("", normalized)

    result: list[str] = []
    prose_lines: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush_prose() -> None:
        if prose_lines:
            result.append(_join_prose_lines(prose_lines))
            prose_lines.clear()

    for raw_line in normalized.split("\n"):
        fence_match = _FENCE.match(raw_line)
        if in_fence:
            result.append(raw_line.rstrip())
            if fence_match and fence_match.group(1)[0] == fence_marker:
                in_fence = False
                fence_marker = ""
            continue

        if fence_match:
            flush_prose()
            result.append(raw_line.strip())
            in_fence = True
            fence_marker = fence_match.group(1)[0]
            continue

        if raw_line.startswith(("    ", "\t")):
            flush_prose()
            result.append(raw_line.rstrip())
            continue

        line = _HORIZONTAL_WHITESPACE.sub(" ", raw_line).strip()
        if not line:
            flush_prose()
            if result and result[-1] != "":
                result.append("")
            continue

        is_markdown_block = bool(_MARKDOWN_BLOCK.match(raw_line)) or "|" in line
        if is_markdown_block:
            flush_prose()
            result.append(line)
            continue

        prose_lines.append(line)

    flush_prose()
    while result and result[-1] == "":
        result.pop()
    return "\n".join(result)


cookies = {
    'cookieCityId': '440100',
    'fvlid': '1789825362012jV3TruAOIc',
    'sessionip': '45.153.5.203',
    'sessionid': 'ADD5E9DF-4D26-4A91-B712-F2462762FF41%7C%7C2026-09-19+21%3A42%3A43.782%7C%7Ccn.bing.com',
    'autoid': 'a4b4ccfc791e271a7ec5af01dc31cbb7',
    'area': '719999',
    '__ah_uuid_ng': 'c_ADD5E9DF-4D26-4A91-B712-F2462762FF41',
    '_ac': 'YQnC9kvG_7MlzzdbDcAmtilpYlcbW4GNLs3MiEa6s-eb30npz39U',
    'ahpvno': '1',
    'ahrlid': '1790430838393Mpbpzrf6CR-1790430839227',
    '.thumbcache_16ba43f8aeefb62e85e0838b29118ea3': 'dPbpjSpLRQyl8ZBFHoLd5q8kpn2vWTLvZmUVkAv49CZoM6dtiAywHsZ1jJx02+XbALZeJ20o2sgwayYTaqnKPg%3D%3D',
    'v_no': '1',
    'visit_info_ad': 'ADD5E9DF-4D26-4A91-B712-F2462762FF41||A623A2C0-1982-4F17-B852-028CD3DB1F13||-1||-1||1',
    'ref': 'cn.bing.com%7C0%7C0%7Cwww.bing.com%7C2026-09-26+21%3A54%3A00.102%7C2026-09-19+21%3A42%3A43.782',
    'sessionvid': 'A623A2C0-1982-4F17-B852-028CD3DB1F13',
}

headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'DNT': '1',
    'Pragma': 'no-cache',
    'Referer': 'https://cn.bing.com/',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'cross-site',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36 Edg/153.0.0.0',
    'sec-ch-ua': '"Microsoft Edge";v="153", "Not_A Brand";v="8", "Chromium";v="153"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    # 'Cookie': 'cookieCityId=440100; fvlid=1789825362012jV3TruAOIc; sessionip=45.153.5.203; sessionid=ADD5E9DF-4D26-4A91-B712-F2462762FF41%7C%7C2026-09-19+21%3A42%3A43.782%7C%7Ccn.bing.com; autoid=a4b4ccfc791e271a7ec5af01dc31cbb7; area=719999; __ah_uuid_ng=c_ADD5E9DF-4D26-4A91-B712-F2462762FF41; _ac=YQnC9kvG_7MlzzdbDcAmtilpYlcbW4GNLs3MiEa6s-eb30npz39U; ahpvno=1; ahrlid=1790430838393Mpbpzrf6CR-1790430839227; .thumbcache_16ba43f8aeefb62e85e0838b29118ea3=dPbpjSpLRQyl8ZBFHoLd5q8kpn2vWTLvZmUVkAv49CZoM6dtiAywHsZ1jJx02+XbALZeJ20o2sgwayYTaqnKPg%3D%3D; v_no=1; visit_info_ad=ADD5E9DF-4D26-4A91-B712-F2462762FF41||A623A2C0-1982-4F17-B852-028CD3DB1F13||-1||-1||1; ref=cn.bing.com%7C0%7C0%7Cwww.bing.com%7C2026-09-26+21%3A54%3A00.102%7C2026-09-19+21%3A42%3A43.782; sessionvid=A623A2C0-1982-4F17-B852-028CD3DB1F13',
}


@tool(policy=ToolPolicy(parallel_safe=True))
class GetUrlTool(BaseTool):
    """抓取 URL 正文，清洗后以 Markdown 返回。"""

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
            cookies=cookies,
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()

        markdown = trafilatura.extract(
            response.text,
            url=url,
            output_format="markdown",
            include_comments=False,
            include_links=True,
            include_images=False,
            include_tables=True,
            favor_precision=True,
            deduplicate=True,
        )

        cleaned_markdown = clean_extracted_markdown(markdown or "")
        if len(cleaned_markdown) < 100:
            raise ValueError("没有提取到有效正文，响应可能是验证页或异常页")

        return cleaned_markdown


if __name__ == "__main__":
    async def main():
        tool = GetUrlTool()

        content = await tool.execute(
            url="https://club.autohome.com.cn/bbs/thread/aa784eeb6d752725/10805490-1.html"
        )


    asyncio.run(main())
