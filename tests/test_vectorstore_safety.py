"""Safety tests for Pinecone index dimension handling.

These tests stub the external SDK so they run without credentials or network access.
"""
import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def load_vectorstore_module():
    pinecone = types.ModuleType("pinecone")
    pinecone.Pinecone = MagicMock()
    pinecone.ServerlessSpec = MagicMock()
    lc_openai = types.ModuleType("langchain_openai")
    lc_openai.OpenAIEmbeddings = MagicMock()
    lc_pinecone = types.ModuleType("langchain_pinecone")
    lc_pinecone.PineconeVectorStore = MagicMock()
    with patch.dict(sys.modules, {
        "pinecone": pinecone,
        "langchain_openai": lc_openai,
        "langchain_pinecone": lc_pinecone,
    }):
        sys.modules.pop("app.rag.vectorstore", None)
        module = importlib.import_module("app.rag.vectorstore")
    return module, pinecone


class VectorstoreSafetyTests(unittest.TestCase):
    def setUp(self):
        self.module, self.pinecone = load_vectorstore_module()
        self.settings = self.module.settings
        self.settings.pinecone_api_key = "test-key"
        self.settings.pinecone_index_name = "existing-index"
        self.settings.embedding_model = "text-embedding-3-small"

    def test_dimension_mismatch_fails_without_deleting_index(self):
        client = self.pinecone.Pinecone.return_value
        client.list_indexes.return_value = [{"name": "existing-index"}]
        client.describe_index.return_value = {"dimension": 384}

        with self.assertRaisesRegex(RuntimeError, "NOT deleted or modified"):
            self.module.ensure_index()

        client.delete_index.assert_not_called()
        client.create_index.assert_not_called()
        client.Index.assert_not_called()

    def test_matching_dimension_reuses_existing_index(self):
        client = self.pinecone.Pinecone.return_value
        client.list_indexes.return_value = [{"name": "existing-index"}]
        client.describe_index.return_value = {"dimension": 1536}
        expected = object()
        client.Index.return_value = expected

        result = self.module.ensure_index()

        self.assertIs(result, expected)
        client.delete_index.assert_not_called()
        client.create_index.assert_not_called()


    def test_retriever_filters_to_active_versions_and_legacy_vectors(self):
        store = MagicMock()
        self.module.get_vectorstore = MagicMock(return_value=store)
        registry = MagicMock()
        registry.get_active_version_ids.return_value = ["version-current"]
        with patch("app.services.document_registry.get_document_registry", return_value=registry):
            self.module.get_retriever()
        kwargs = store.as_retriever.call_args.kwargs
        self.assertEqual(kwargs["search_kwargs"]["k"], self.settings.top_k)
        self.assertEqual(kwargs["search_kwargs"]["filter"], {"$or": [
            {"version_id": {"$in": ["version-current"]}},
            {"version_id": {"$exists": False}},
        ]})

    def test_retriever_with_no_registry_versions_keeps_legacy_vectors(self):
        store = MagicMock()
        self.module.get_vectorstore = MagicMock(return_value=store)
        registry = MagicMock()
        registry.get_active_version_ids.return_value = []
        with patch("app.services.document_registry.get_document_registry", return_value=registry):
            self.module.get_retriever()
        kwargs = store.as_retriever.call_args.kwargs
        self.assertEqual(kwargs["search_kwargs"]["filter"], {
            "$or": [{"version_id": {"$exists": False}}]
        })

    def test_unknown_existing_dimension_fails_closed(self):
        client = self.pinecone.Pinecone.return_value
        client.list_indexes.return_value = [{"name": "existing-index"}]
        client.describe_index.return_value = {}

        with self.assertRaisesRegex(RuntimeError, "Cannot verify the dimension"):
            self.module.ensure_index()

        client.delete_index.assert_not_called()
        client.create_index.assert_not_called()


if __name__ == "__main__":
    unittest.main()
