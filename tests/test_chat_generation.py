import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from rag_engine.query import RAGPipeline

CHUNKS = [{"id": "chunk-a", "document": "Python", "metadata": {"source": "resume.txt"}}]
VALID = '{"answer":"Python [1].","context_sufficient":true,"cited_chunk_ids":["chunk-a"]}'


class ChatGenerationTests(unittest.TestCase):
    def test_gemini_uses_system_instruction_and_typed_json_output(self):
        pipeline = RAGPipeline()
        pipeline._client = MagicMock()
        pipeline._client.models.generate_content.return_value = SimpleNamespace(text=VALID)
        result = pipeline.run("What skill?", CHUNKS)
        self.assertEqual(result.cited_chunk_ids, ["chunk-a"])
        config = pipeline._client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(config.response_json_schema["additionalProperties"], False)
        self.assertIn("History is for understanding", config.system_instruction)

    def test_groq_fallback_is_used_after_gemini_failure(self):
        pipeline = RAGPipeline()
        pipeline._client = MagicMock()
        pipeline._client.models.generate_content.side_effect = RuntimeError("Unavailable")
        pipeline._groq_client = MagicMock()
        pipeline._groq_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=VALID))],
        )
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-placeholder"}):
            result = pipeline.run("What skill?", CHUNKS)
        self.assertEqual(result.answer, "Python [1].")
        args = pipeline._groq_client.chat.completions.create.call_args.kwargs
        self.assertEqual(args["messages"][0]["role"], "system")
        self.assertEqual(args["response_format"], {"type": "json_object"})

    def test_invalid_provider_output_is_not_displayed_as_an_answer(self):
        pipeline = RAGPipeline()
        pipeline._client = MagicMock()
        pipeline._client.models.generate_content.return_value = SimpleNamespace(text="not JSON")
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}):
            with self.assertRaises(RuntimeError):
                pipeline.run("What skill?", CHUNKS)


if __name__ == "__main__":
    unittest.main()
