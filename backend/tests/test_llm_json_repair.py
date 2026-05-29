import json

import pytest

from app.utils.llm_client import parse_llm_json_response


def test_parse_llm_json_response_accepts_valid_json():
    payload = {"entity_types": [], "edge_types": [], "analysis_summary": "ok"}
    assert parse_llm_json_response(json.dumps(payload)) == payload


def test_parse_llm_json_response_strips_markdown_fences():
    payload = {"entity_types": [], "edge_types": [], "analysis_summary": "ok"}
    wrapped = f"```json\n{json.dumps(payload)}\n```"
    assert parse_llm_json_response(wrapped) == payload


def test_parse_llm_json_response_repairs_trailing_garbage_after_last_field():
    corrupted = """
{
    "entity_types": [
        {
            "name": "Executive",
            "description": "Corporate leaders.",
            "attributes": [],
            "examples": ["CEO"]
        }
    ],
    "edge_types": [
        {
            "name": "WORKS_FOR",
            "description": "Employment relationship.",
            "source_targets": [{"source": "Executive", "target": "Company"}],
            "attributes": []
        }
    ],
    "analysis_summary": "Models executives and companies for sentiment simulation."
)."
and investing)."
}
"""
    parsed = parse_llm_json_response(corrupted)
    assert parsed["entity_types"][0]["name"] == "Executive"
    assert parsed["edge_types"][0]["name"] == "WORKS_FOR"
    assert "sentiment simulation" in parsed["analysis_summary"]


def test_parse_llm_json_response_closes_truncated_json():
    truncated = """
{
    "entity_types": [
        {"name": "Person", "description": "Fallback person type", "attributes": [], "examples": []}
    ],
    "edge_types": [],
    "analysis_summary": "Partial summary without closing braces
"""
    parsed = parse_llm_json_response(truncated)
    assert parsed["entity_types"][0]["name"] == "Person"
    assert parsed["analysis_summary"].startswith("Partial summary")


def test_parse_llm_json_response_raises_on_unrecoverable_json():
    with pytest.raises(ValueError, match="LLM returned invalid JSON"):
        parse_llm_json_response("not json at all")


def test_parse_llm_json_response_repairs_trailing_garbage_with_multiple_quotes():
    corrupted = """
{
    "entity_types": [],
    "edge_types": [],
    "analysis_summary": "This ontology is designed to model..."
    without losing structural integrity."
    integrity."
    integrity."
}
"""
    parsed = parse_llm_json_response(corrupted)
    assert parsed["entity_types"] == []
    assert parsed["edge_types"] == []
    assert parsed["analysis_summary"] == "This ontology is designed to model..."

