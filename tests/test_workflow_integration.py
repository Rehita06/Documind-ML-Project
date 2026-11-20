"""Offline tests for LangGraph intent routing and the new workflow branches."""
from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import patch


def _install_workflow_import_stubs():
    # The application dependencies are pinned in requirements.txt, but are not
    # installed in this offline test environment. These stubs exercise the
    # workflow functions and graph wiring without pretending to test providers.
    core = types.ModuleType("langchain_core")
    documents = types.ModuleType("langchain_core.documents")
    documents.Document = type("Document", (), {})
    core.documents = documents

    openai = types.ModuleType("langchain_openai")
    openai.ChatOpenAI = type("ChatOpenAI", (), {})
    tavily = types.ModuleType("langchain_tavily")
    tavily.TavilySearch = type("TavilySearch", (), {})

    graph_mod = types.ModuleType("langgraph.graph")
    graph_mod.START = "__start__"
    graph_mod.END = "__end__"

    class FakeGraph:
        def __init__(self, _state):
            self.nodes = {}
            self.edges = []
            self.conditional = []

        def add_node(self, name, fn):
            self.nodes[name] = fn

        def add_edge(self, start, end):
            self.edges.append((start, end))

        def add_conditional_edges(self, start, fn, mapping):
            self.conditional.append((start, fn, mapping))

        def compile(self):
            return self

        def invoke(self, state):
            return state

    graph_mod.StateGraph = FakeGraph
    langgraph = types.ModuleType("langgraph")
    langgraph.graph = graph_mod

    vectorstore = types.ModuleType("app.rag.vectorstore")
    vectorstore.get_retriever = lambda: None

    sys.modules.update({
        "langchain_core": core,
        "langchain_core.documents": documents,
        "langchain_openai": openai,
        "langchain_tavily": tavily,
        "langgraph": langgraph,
        "langgraph.graph": graph_mod,
        "app.rag.vectorstore": vectorstore,
    })


_install_workflow_import_stubs()
workflow = importlib.import_module("app.rag.workflow")


class FakeRouter:
    def __init__(self, route):
        self.route = route

    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, _prompt):
        return types.SimpleNamespace(route=self.route)


class WorkflowIntegrationTests(unittest.TestCase):
    def _state(self, **updates):
        state = {
            "question": "compare these policies",
            "trace": [],
            "source_used": "",
            "document_version_ids": [],
            "old_version_id": None,
            "new_version_id": None,
        }
        state.update(updates)
        return state

    def test_explicit_intent_bypasses_llm(self):
        with patch.object(workflow, "llm", side_effect=AssertionError("LLM should not run")):
            result = workflow.route_question(self._state(requested_intent="verify_documents"))
        self.assertEqual(result["source_used"], "verify_documents")
        self.assertIn("explicit request", result["trace"][-1])

    def test_router_classifies_verification(self):
        with patch.object(workflow, "llm", return_value=FakeRouter("verify_documents")):
            result = workflow.route_question(self._state())
        self.assertEqual(result["source_used"], "verify_documents")

    def test_router_classifies_version_comparison(self):
        with patch.object(workflow, "llm", return_value=FakeRouter("compare_versions")):
            result = workflow.route_question(self._state())
        self.assertEqual(result["source_used"], "compare_versions")

    def test_router_failure_falls_back_to_private_kb(self):
        with patch.object(workflow, "llm", side_effect=RuntimeError("provider unavailable")):
            result = workflow.route_question(self._state())
        self.assertEqual(result["source_used"], "kb")
        self.assertIn("safe KB fallback", result["trace"][-1])

    def test_route_mapping_includes_all_four_paths(self):
        expected = {
            "kb": "retrieve_kb",
            "direct": "direct_answer",
            "verify_documents": "verify_documents",
            "compare_versions": "compare_versions",
        }
        for route, node in expected.items():
            self.assertEqual(workflow.route_after_router({"source_used": route}), node)

    def test_graph_registers_new_nodes_and_terminal_edges(self):
        graph = workflow.build_graph()
        self.assertIn("verify_documents", graph.nodes)
        self.assertIn("compare_versions", graph.nodes)
        self.assertIn(("verify_documents", "__end__"), graph.edges)
        self.assertIn(("compare_versions", "__end__"), graph.edges)
        start_routes = next(item[2] for item in graph.conditional if item[0] == "route_question")
        self.assertEqual(start_routes["verify_documents"], "verify_documents")
        self.assertEqual(start_routes["compare_versions"], "compare_versions")

    def test_verification_without_selected_versions_is_actionable(self):
        result = workflow.verify_documents(self._state())
        self.assertEqual(result["source_used"], "document_verification")
        self.assertIsNone(result["verification_report"])
        self.assertIn("at least two", result["answer"])

    def test_comparison_without_selected_versions_is_actionable(self):
        result = workflow.compare_versions(self._state())
        self.assertEqual(result["source_used"], "version_comparison")
        self.assertIsNone(result["comparison_report"])
        self.assertIn("old version ID", result["answer"])

    def test_ask_preserves_original_question_and_accepts_workflow_selection(self):
        result = workflow.ask(
            "verify these documents", intent="verify_documents",
            document_version_ids=["v1", "v2"],
        )
        self.assertEqual(result["question"], "verify these documents")
        self.assertEqual(result["requested_intent"], "verify_documents")
        self.assertEqual(result["document_version_ids"], ["v1", "v2"])


if __name__ == "__main__":
    unittest.main()

class WorkflowServiceBranchTests(unittest.TestCase):
    def _state(self, **updates):
        state = {
            "question": "run document analysis", "trace": [], "source_used": "",
            "document_version_ids": [], "old_version_id": None, "new_version_id": None,
        }
        state.update(updates)
        return state

    def test_verification_branch_returns_report_citations_and_model_trace(self):
        fake_report = {
            "run_id": "run-1", "status": "complete", "model_version": "ann-test",
            "summary": {"evaluated_pairs": 1, "label_counts": {
                "CONTRADICTION": 1, "CONSISTENT": 0, "UNCERTAIN": 0},
                "human_review_required": 1},
            "findings": [{
                "source_a": {"logical_name": "Policy A", "version_id": "va", "location": "line:1", "chunk_id": "ca"},
                "source_b": {"logical_name": "Policy B", "version_id": "vb", "location": "line:2", "chunk_id": "cb"},
            }],
        }
        fake_module = types.ModuleType("app.services.verification")
        fake_module.CrossDocumentVerificationService = lambda: types.SimpleNamespace(
            verify_versions=lambda ids: fake_report if ids == ["va", "vb"] else self.fail("unexpected IDs")
        )
        with patch.dict(sys.modules, {"app.services.verification": fake_module}):
            result = workflow.verify_documents(self._state(document_version_ids=["va", "vb"]))
        self.assertEqual(result["verification_report"], fake_report)
        self.assertEqual(result["source_used"], "document_verification")
        self.assertEqual(len(result["citations"]), 2)
        self.assertIn("ANN ann-test", result["trace"][-2])
        self.assertIn("1 possible contradictions", result["answer"])

    def test_comparison_branch_returns_change_records_and_source_versions(self):
        fake_report = {
            "document_id": "doc-1", "alignment_method": "local_token_similarity_fallback",
            "counts": {"ADDED": 1, "DELETED": 0, "MODIFIED": 1, "UNCHANGED": 3},
            "old_version": {"logical_name": "VPN Policy", "version_id": "old", "version_number": 1},
            "new_version": {"logical_name": "VPN Policy", "version_id": "new", "version_number": 2},
            "changes": [{"change_type": "MODIFIED", "old_text": "old", "new_text": "new"}],
        }
        change_module = types.ModuleType("app.services.change_detection")
        change_module.compare_document_versions = lambda registry, old, new: fake_report
        registry_module = types.ModuleType("app.services.document_registry")
        registry_module.get_document_registry = lambda: object()
        audit_module = types.ModuleType("app.services.audit")
        audit_calls = []
        audit_module.write_document_audit = lambda *args: audit_calls.append(args)
        with patch.dict(sys.modules, {
            "app.services.change_detection": change_module,
            "app.services.document_registry": registry_module,
            "app.services.audit": audit_module,
        }):
            result = workflow.compare_versions(self._state(old_version_id="old", new_version_id="new"))
        self.assertEqual(result["comparison_report"], fake_report)
        self.assertEqual(result["source_used"], "version_comparison")
        self.assertEqual(len(result["citations"]), 2)
        self.assertEqual(len(audit_calls), 1)
        self.assertIn("1 added", result["answer"])
        self.assertIn("not automatically a policy contradiction", result["answer"])
