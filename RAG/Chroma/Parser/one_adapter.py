from typing import Any, Dict, List


class OneParser:
    """将 RAGFlow 旧式 ``one.chunk`` 适配为统一的 Parser 接口。"""

    def parse(self, **kwargs: Any) -> List[Dict[str, Any]]:
        from .one import chunk

        return chunk(**kwargs)
