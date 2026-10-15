from app.hybrid_retriever import _where_clause


def test_session_filter_does_not_include_global_or_other_sessions():
    assert _where_clause(None, "session-a") == {"session_id": "session-a"}


def test_session_filter_and_document_filter_are_combined():
    assert _where_clause(["report.pdf"], "session-a") == {
        "$and": [
            {"session_id": "session-a"},
            {"document_name": "report.pdf"},
        ]
    }


def test_multiple_document_names_use_in_filter():
    assert _where_clause(["a.pdf", "b.pdf"], "session-a") == {
        "$and": [
            {"session_id": "session-a"},
            {"document_name": {"$in": ["a.pdf", "b.pdf"]}},
        ]
    }
