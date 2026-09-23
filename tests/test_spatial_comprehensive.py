"""Behavioral coverage for deterministic scene geometry and extraction."""

import json
from unittest.mock import AsyncMock

import pytest

from fieldwork.spatial import (
    SpatialAnalyzer,
    SpatialEntity,
    SpatialRelation,
    SpatialScene,
)

pytestmark = pytest.mark.unit


def test_bulk_distance_handles_existing_and_missing_geometry():
    scene = SpatialScene()
    scene.add_entity(SpatialEntity("a", "worker", (0.0, 0.0)))
    scene.add_entity(SpatialEntity("b", "forklift", (3.0, 4.0)))
    scene.add_entity(SpatialEntity("missing", "crate"))
    computed = SpatialRelation("a", "near", "b")
    preserved = SpatialRelation("a", "near", "b", 9.0)
    unavailable = SpatialRelation("a", "near", "missing")
    for relation in (computed, preserved, unavailable):
        scene.add_relation(relation)

    assert scene.compute_distance("unknown", "b") is None
    assert scene.compute_distance("a", "missing") is None
    scene.compute_all_distances()

    assert computed.distance == 5.0
    assert preserved.distance == 9.0
    assert unavailable.distance is None


def test_query_near_ignores_self_far_and_unpositioned_entities():
    scene = SpatialScene()
    scene.add_entity(SpatialEntity("origin", "worker", (0.0, 0.0)))
    scene.add_entity(SpatialEntity("near", "forklift", (1.0, 0.0)))
    scene.add_entity(SpatialEntity("far", "forklift", (20.0, 0.0)))
    scene.add_entity(SpatialEntity("unknown", "crate"))

    assert [entity.id for entity in scene.query_near("origin", 2.0)] == ["near"]


def test_constraint_checks_cover_ppe_and_distance_guardrails():
    scene = SpatialScene(violations=["stale"])
    scene.add_entity(
        SpatialEntity(
            "worker",
            "worker",
            attributes={
                "wearing_ppe": False,
                "hard_hat": False,
                "safety_vest": False,
            },
            zone="dock",
        )
    )
    scene.add_entity(SpatialEntity("forklift", "forklift"))
    scene.add_entity(SpatialEntity("cone", "cone"))
    scene.add_relation(SpatialRelation("worker", "near", "forklift", 1.0))
    scene.add_relation(SpatialRelation("worker", "near", "cone", 1.0))
    scene.add_relation(SpatialRelation("forklift", "near", "worker", 1.0))
    scene.add_relation(SpatialRelation("worker", "near", "missing", 1.0))
    scene.add_relation(SpatialRelation("worker", "near", "forklift", None))
    scene.add_relation(SpatialRelation("worker", "near", "forklift", 10.0))
    scene.safety_rules = [
        "Workers must wear PPE, hard hat, and safety vest",
        "Maintain distance from forklifts",
        "Maintain 3 meters from forklifts",
        "Keep aisles clear",
    ]

    violations = scene.check_constraints()

    assert "stale" not in violations
    assert any("missing PPE" in violation for violation in violations)
    assert any("missing hard hat" in violation for violation in violations)
    assert any("missing safety vest" in violation for violation in violations)
    assert any("1.0m < 3.0m" in violation for violation in violations)


def test_fact_sheet_covers_relationship_zones_attributes_and_status():
    scene = SpatialScene()
    scene.add_entity(SpatialEntity("worker", "worker", attributes={"ppe": True}))
    scene.add_relation(SpatialRelation("worker", "inside", "dock"))
    scene.zones = {"dock": {"type": "hazard"}, "yard": {}}
    scene.violations = ["blocked exit"]

    sheet = scene.to_fact_sheet()

    assert "ppe=True" in sheet
    assert "worker inside dock" in sheet
    assert "dock: hazard" in sheet
    assert "yard: unknown" in sheet
    assert "VIOLATION: blocked exit" in sheet

    safe = SpatialScene(safety_rules=["Keep aisles clear"])
    assert "No violations detected" in safe.to_fact_sheet()


@pytest.mark.asyncio
async def test_analyzer_builds_complete_scene_and_defaults():
    llm = AsyncMock()
    llm.generate.return_value = json.dumps(
        {
            "entities": [
                {
                    "id": "worker",
                    "label": "worker",
                    "position_x": 0,
                    "position_y": 0,
                    "zone": "dock",
                    "attributes": {"wearing_ppe": False},
                },
                {
                    "id": "forklift",
                    "label": "forklift",
                    "position_x": 3,
                    "position_y": 4,
                },
                {"id": "partial", "position_x": 1},
                {},
            ],
            "relations": [
                {"subject": "worker", "object": "forklift"},
                {},
            ],
            "zones": [
                {"name": "dock", "type": "hazard"},
                {},
            ],
            "safety_rules": [
                "Workers must wear PPE",
                "Maintain 6 meters from forklifts",
            ],
        }
    )
    analyzer = SpatialAnalyzer(llm)

    scene = await analyzer.build_scene("query", ["camera one", "camera two"])

    assert scene.entity_count == 4
    assert scene.entities["worker"].position == (0.0, 0.0)
    assert scene.entities["partial"].position is None
    assert scene.entities["e3"].label == "unknown"
    assert scene.relations[0].distance == 5.0
    assert scene.relations[1].predicate == "near"
    assert scene.zones["dock"] == {"type": "hazard"}
    assert scene.zones["unknown"] == {"type": "general"}
    assert scene.violation_count == 2
    prompt = llm.generate.await_args.args[0]
    assert "camera one\n\ncamera two" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["not json", RuntimeError("provider down")])
async def test_analyzer_failure_returns_empty_scene(failure):
    llm = AsyncMock()
    if isinstance(failure, Exception):
        llm.generate.side_effect = failure
    else:
        llm.generate.return_value = failure

    scene = await SpatialAnalyzer(llm).build_scene("query", ["context"])

    assert scene.entity_count == 0
    assert scene.relations == []
