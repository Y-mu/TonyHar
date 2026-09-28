"""与内容来源无关的文本切块和入库服务。"""

from collections.abc import Mapping
from typing import Any

from .document_service import DocumentService
from .parser import RawDocument
from .pipeline_context import PipelineContext
from .splitter import TextSplitter


class TextIngestionService:
    """把已经解析好的正文切块并写入知识库。"""

    def __init__(
        self,
        splitter: TextSplitter,
        document_service: DocumentService,
    ) -> None:
        self.splitter = splitter
        self.document_service = document_service

    def ingest(
        self,
        text: str,
        *,
        document_id: str,
        filename: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> PipelineContext:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("待入库文本不能为空")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id 不能为空")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename 不能为空")

        document_metadata = dict(metadata or {})
        context = PipelineContext(
            document_id=document_id,
            filename=filename,
            metadata=document_metadata,
        )
        try:
            context.raw_document = RawDocument(
                text=text,
                document_id=document_id,
                filename=filename,
                metadata=document_metadata,
            )
            context.update_progress(0.4)
            chunks = self.splitter.split(context.raw_document)
            if not chunks:
                raise ValueError("文本没有生成有效切块")
            context.update_progress(0.7)
            context.chunks = self.document_service.index_chunks(chunks)
            context.update_progress(1.0)
            return context
        except Exception as exc:
            context.add_error(str(exc))
            raise
