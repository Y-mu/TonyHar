"""文档处理流程调度器。"""
from .document_service import DocumentService
from .pipeline_context import PipelineContext


class DocumentScheduler:
    def __init__(self, parser, splitter, document_service: DocumentService):
        self.parser = parser
        self.splitter = splitter
        self.document_service = document_service

    def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            if ctx.binary is None:
                raise ValueError("ctx.binary 不能为空")
            ctx.update_progress(0.1)
            ctx.raw_document = self.parser.parse(ctx.filename, ctx.binary)
            # 调度上下文是任务身份的唯一来源，覆盖 Parser 的临时标识。
            ctx.raw_document.document_id = ctx.document_id
            ctx.raw_document.metadata.update(ctx.metadata)
            ctx.update_progress(0.4)
            ctx.chunks = self.splitter.split(ctx.raw_document)
            ctx.update_progress(0.7)
            # 将业务 metadata 回写到上下文，供后续阶段继续使用。
            ctx.chunks = [
                self.document_service.prepare_chunk(chunk)
                for chunk in ctx.chunks
            ]
            self.document_service.index_chunks(ctx.chunks)
            ctx.update_progress(1.0)
            return ctx
        except Exception as exc:
            ctx.add_error(str(exc))
            raise
