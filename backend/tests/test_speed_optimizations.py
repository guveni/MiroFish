from app.services.graphiti_graph_builder import _in_batch_progress_ratio
from app.services.oasis_profile_generator import OasisProfileGenerator
from app.services.zep_entity_reader import EntityNode, ZepEntityReader


def test_in_batch_progress_ratio_interpolates_and_caps():
    assert _in_batch_progress_ratio(0.0, 0.1, 0.0, 90) == 0.0
    assert _in_batch_progress_ratio(0.0, 0.1, 45.0, 90) == 0.05
    assert _in_batch_progress_ratio(0.0, 0.1, 90.0, 90) == 0.095
    assert _in_batch_progress_ratio(0.2, 0.2, 10.0, 90) == 0.2


def test_filter_defined_entities_indexes_edges_without_changing_enrichment(monkeypatch):
    reader = object.__new__(ZepEntityReader)
    nodes = [
        {
            "uuid": "n1",
            "name": "Alice",
            "labels": ["Entity", "Person"],
            "summary": "A person",
            "attributes": {},
        },
        {
            "uuid": "n2",
            "name": "University",
            "labels": ["Entity", "University"],
            "summary": "A school",
            "attributes": {},
        },
        {
            "uuid": "n3",
            "name": "Plain",
            "labels": ["Entity"],
            "summary": "Default node",
            "attributes": {},
        },
    ]
    edges = [
        {
            "uuid": "e1",
            "name": "ATTENDS",
            "fact": "Alice attends University",
            "source_node_uuid": "n1",
            "target_node_uuid": "n2",
            "attributes": {},
        },
        {
            "uuid": "e2",
            "name": "MENTIONS",
            "fact": "University mentions Alice",
            "source_node_uuid": "n2",
            "target_node_uuid": "n1",
            "attributes": {},
        },
    ]

    monkeypatch.setattr(reader, "get_all_nodes", lambda graph_id: nodes)
    monkeypatch.setattr(reader, "get_all_edges", lambda graph_id: edges)

    filtered = reader.filter_defined_entities("graph-1", enrich_with_edges=True)

    alice = next(entity for entity in filtered.entities if entity.uuid == "n1")
    assert filtered.total_count == 3
    assert filtered.filtered_count == 2
    assert {edge["direction"] for edge in alice.related_edges} == {"incoming", "outgoing"}
    assert {node["uuid"] for node in alice.related_nodes} == {"n2"}


def test_profile_context_skips_search_when_direct_context_is_sufficient(monkeypatch):
    generator = object.__new__(OasisProfileGenerator)
    generator.graph_tools = object()
    generator.graph_id = "graph-1"

    entity = EntityNode(
        uuid="n1",
        name="Alice",
        labels=["Entity", "Person"],
        summary="A person",
        attributes={},
        related_edges=[
            {"fact": "Fact 1", "edge_name": "REL", "direction": "outgoing"},
            {"fact": "Fact 2", "edge_name": "REL", "direction": "outgoing"},
        ],
        related_nodes=[
            {
                "uuid": "n2",
                "name": "Related",
                "labels": ["Entity", "Organization"],
                "summary": "Fact 3",
            }
        ],
    )

    def fail_search(entity):
        raise AssertionError("Search should be skipped for rich direct context")

    monkeypatch.setattr(generator, "_search_zep_for_entity", fail_search)

    context = generator._build_entity_context(entity)

    assert "Fact 1" in context
    assert "Fact 2" in context
    assert "Related" in context


def test_profile_context_searches_when_direct_context_is_sparse(monkeypatch):
    generator = object.__new__(OasisProfileGenerator)
    generator.graph_tools = object()
    generator.graph_id = "graph-1"

    entity = EntityNode(
        uuid="n1",
        name="Alice",
        labels=["Entity", "Person"],
        summary="A person",
        attributes={},
        related_edges=[],
        related_nodes=[],
    )

    monkeypatch.setattr(
        generator,
        "_search_zep_for_entity",
        lambda entity: {
            "facts": ["Search fact"],
            "node_summaries": ["Search node"],
            "context": "",
        },
    )

    context = generator._build_entity_context(entity)

    assert "Search fact" in context
    assert "Search node" in context
