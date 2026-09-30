import tempfile
import unittest
from pathlib import Path

from tonyhar.agent.llm import LLMCompleted, LLMResponse
from tonyhar.agent.runnables import AgentResult, AgentState
from tonyhar.conversation import (
    ChatService,
    ContextBuilder,
    ContextPolicy,
    ExtractiveMemoryCompactor,
    LLMMemoryCompactor,
    MemorySnapshot,
    InMemorySessionStore,
    SQLiteSessionStore,
    SessionConflictError,
    Session,
    TokenCounter,
    TurnStatus,
)
from tonyhar.resilience import Deadline


class MemoryLLM:
    def __init__(self, content: str):
        self.content = content

    async def stream(self, messages, tools, *, deadline):
        del messages, tools, deadline
        yield LLMCompleted(LLMResponse(content=self.content))

    async def aclose(self):
        return None


class EchoAgent:
    async def invoke(self, context):
        context.transition_to(AgentState.DISPATCHING)
        context.add_message("user", context.user_input)
        context.add_message("assistant", f"answer:{context.user_input}")
        context.transition_to(AgentState.COMPLETED)
        return AgentResult(
            run_id=context.run_id,
            session_id=context.session_id,
            success=True,
            answer=f"answer:{context.user_input}",
        )

    async def stream(self, context):
        await self.invoke(context)
        if False:
            yield None

    async def aclose(self):
        return None


class CountingCompactor(ExtractiveMemoryCompactor):
    def __init__(self, token_counter):
        super().__init__(token_counter, max_tokens=200)
        self.calls = 0

    async def compact(self, snapshot, messages, *, deadline):
        self.calls += 1
        return await super().compact(
            snapshot,
            messages,
            deadline=deadline,
        )


class ShortTermMemoryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.counter = TokenCounter()

    def test_context_excludes_failed_turns_and_includes_memory(self):
        session = Session.create("session", system_prompt="system")
        session.memory = MemorySnapshot(
            summary="用户正在实现短期记忆",
            task_state={"current_goal": "实现 ContextBuilder"},
        )
        session.append_turn(
            [
                {"role": "user", "content": "成功问题"},
                {"role": "assistant", "content": "成功回答"},
            ],
            status=TurnStatus.COMPLETED,
            turn_id="completed",
        )
        session.append_turn(
            [{"role": "user", "content": "失败问题"}],
            status=TurnStatus.FAILED,
            turn_id="failed",
        )

        messages = ContextBuilder(self.counter).build(session, "继续")
        contents = [message["content"] for message in messages]

        self.assertEqual(messages[0], {"role": "system", "content": "system"})
        self.assertIn("session_memory", messages[1]["content"])
        self.assertIn("成功问题", contents)
        self.assertNotIn("失败问题", contents)

    def test_oversized_turn_uses_semantic_view_without_tool_chain(self):
        session = Session.create("session")
        session.append_turn(
            [
                {"role": "user", "content": "分析这个大结果" * 200},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call-1"}],
                },
                {
                    "role": "tool",
                    "content": "工具正文" * 2_000,
                    "tool_call_id": "call-1",
                },
                {"role": "assistant", "content": "最终结论" * 200},
            ],
            status=TurnStatus.COMPLETED,
        )
        builder = ContextBuilder(
            self.counter,
            ContextPolicy(
                max_context_tokens=300,
                response_reserve_tokens=40,
                tool_schema_reserve_tokens=40,
            ),
        )

        messages = builder.build(session, "继续")

        self.assertTrue(messages)
        self.assertEqual({message["role"] for message in messages}, {
            "user",
            "assistant",
        })
        self.assertNotIn("tool_calls", messages[-1])

    async def test_extractive_compaction_marks_covered_prefix(self):
        session = Session.create("session")
        for index in range(3):
            session.append_turn(
                [
                    {"role": "user", "content": f"问题 {index}"},
                    {"role": "assistant", "content": f"回答 {index}"},
                ],
                status=TurnStatus.COMPLETED,
                turn_id=f"turn-{index}",
            )
        builder = ContextBuilder(
            self.counter,
            ContextPolicy(
                max_context_tokens=400,
                response_reserve_tokens=20,
                tool_schema_reserve_tokens=20,
                minimum_recent_turns=1,
                compaction_batch_tokens=1_000,
            ),
        )
        candidates = builder.compaction_candidates(session)
        snapshot = await ExtractiveMemoryCompactor(
            self.counter,
            max_tokens=200,
        ).compact(
            session.memory,
            candidates,
            deadline=Deadline.after(1),
        )
        session.memory = snapshot

        self.assertEqual(snapshot.version, 1)
        self.assertEqual(snapshot.covered_through_message_id, candidates[-1].id)
        self.assertIn("问题 0", snapshot.summary)
        remaining = builder.build(session, "继续")
        raw_turn_text = "\n".join(
            message["content"]
            for message in remaining
            if message["role"] != "system"
        )
        self.assertNotIn("问题 0", raw_turn_text)
        self.assertIn("问题 2", raw_turn_text)

    async def test_sqlite_store_survives_reopen(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sessions.sqlite3"
            store = SQLiteSessionStore(path)
            session = Session.create("persisted", system_prompt="system")
            session.append_turn(
                [
                    {"role": "user", "content": "记住这一点"},
                    {"role": "assistant", "content": "已经记住"},
                ],
                status=TurnStatus.COMPLETED,
            )
            await store.create(session)
            await store.aclose()

            reopened = SQLiteSessionStore(path)
            loaded = await reopened.get("persisted")
            await reopened.aclose()

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.system_prompt, "system")
            self.assertEqual(
                [message.content for message in loaded.messages],
                ["记住这一点", "已经记住"],
            )

    async def test_sqlite_store_rejects_stale_revision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteSessionStore(Path(temp_dir) / "sessions.sqlite3")
            await store.create(Session.create("session"))
            first = await store.get("session")
            stale = await store.get("session")
            assert first is not None and stale is not None

            first.append_turn(
                [{"role": "user", "content": "first"}],
                status=TurnStatus.COMPLETED,
            )
            await store.save(first)
            stale.append_turn(
                [{"role": "user", "content": "stale"}],
                status=TurnStatus.COMPLETED,
            )
            with self.assertRaises(SessionConflictError):
                await store.save(stale)
            await store.aclose()

    async def test_llm_compactor_produces_structured_memory(self):
        session = Session.create("session")
        session.append_turn(
            [
                {"role": "user", "content": "保持模块化单体"},
                {"role": "assistant", "content": "已经确认"},
            ],
            status=TurnStatus.COMPLETED,
        )
        llm = MemoryLLM(
            '{"summary":"用户要求保持模块化单体",'
            '"task_state":{"current_goal":"完善短期记忆",'
            '"constraints":["不拆微服务"],"decisions":[],'
            '"open_questions":[]}}'
        )

        snapshot = await LLMMemoryCompactor(
            llm,
            self.counter,
            max_tokens=200,
        ).compact(
            session.memory,
            session.messages,
            deadline=Deadline.after(1),
        )

        self.assertEqual(snapshot.version, 1)
        self.assertEqual(snapshot.task_state["current_goal"], "完善短期记忆")
        self.assertEqual(
            snapshot.covered_through_message_id,
            session.messages[-1].id,
        )

    async def test_chat_service_compacts_before_building_context(self):
        store = InMemorySessionStore()
        session = Session.create("session")
        for index in range(5):
            session.append_turn(
                [
                    {"role": "user", "content": f"问题 {index} " * 20},
                    {"role": "assistant", "content": f"回答 {index} " * 20},
                ],
                status=TurnStatus.COMPLETED,
                turn_id=f"turn-{index}",
            )
        await store.create(session)
        policy = ContextPolicy(
            max_context_tokens=400,
            response_reserve_tokens=40,
            tool_schema_reserve_tokens=40,
            minimum_recent_turns=1,
            compact_threshold=0.5,
            compaction_batch_tokens=2_000,
            max_compactions_per_run=2,
        )
        compactor = CountingCompactor(self.counter)
        service = ChatService(
            EchoAgent(),
            store,
            context_builder=ContextBuilder(self.counter, policy),
            memory_compactor=compactor,
        )

        result = await service.invoke("session", "继续")
        saved = await store.get("session")
        await service.aclose()

        self.assertTrue(result.success)
        self.assertGreaterEqual(compactor.calls, 1)
        self.assertIsNotNone(saved)
        assert saved is not None
        self.assertGreaterEqual(saved.memory.version, 1)


if __name__ == "__main__":
    unittest.main()
