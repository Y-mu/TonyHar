import unittest
from types import SimpleNamespace

from core.userIntentrecognizer import (
    DEFAULT_INTENT_ROUTER_CONFIG,
    UserIntentRecognizer,
)


class FakeLayer:
    def __init__(self, choice):
        self.choice = choice
        self.calls = []

    def __call__(self, text, simulate_static):
        self.calls.append((text, simulate_static))
        return self.choice


def recognizer_without_model(choice=None) -> UserIntentRecognizer:
    """绕过昂贵的模型构造，只测试识别器自己的边界逻辑。"""
    recognizer = UserIntentRecognizer.__new__(UserIntentRecognizer)
    recognizer.config = DEFAULT_INTENT_ROUTER_CONFIG
    recognizer.layer = FakeLayer(choice)
    return recognizer


class UserIntentRecognizerTest(unittest.TestCase):
    def test_blank_input_is_rejected_without_calling_router(self):
        recognizer = recognizer_without_model()

        result = recognizer.classify("   ")

        self.assertEqual(result.name, "unknown")
        self.assertEqual(result.matched_by, "validation")
        self.assertEqual(recognizer.layer.calls, [])

    def test_non_string_input_is_rejected(self):
        recognizer = recognizer_without_model()

        result = recognizer.classify(None)  # type: ignore[arg-type]

        self.assertEqual(result.name, "unknown")
        self.assertEqual(result.matched_by, "validation")

    def test_score_is_preserved_and_static_simulation_is_disabled(self):
        choice = SimpleNamespace(name="chat", similarity_score=0.72)
        recognizer = recognizer_without_model(choice)

        result = recognizer.classify("  你好  ")

        self.assertEqual(result.name, "chat")
        self.assertEqual(result.score, 0.72)
        self.assertEqual(recognizer.layer.calls, [("你好", False)])

    def test_failed_threshold_returns_unknown(self):
        choice = SimpleNamespace(name="chat", similarity_score=0.30)
        recognizer = recognizer_without_model(choice)

        result = recognizer.classify("不属于任何已有意图")

        self.assertEqual(result.name, "unknown")
        self.assertEqual(result.score, 0.30)
        self.assertEqual(result.matched_by, "threshold")

    def test_ingest_corpus_contains_filename_and_path_expressions(self):
        utterances = DEFAULT_INTENT_ROUTER_CONFIG.utterances["ingest"]

        self.assertIn("请把 manual.txt 导入知识库", utterances)
        self.assertTrue(any("/path/" in item for item in utterances))


if __name__ == "__main__":
    unittest.main()
