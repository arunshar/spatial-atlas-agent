"""
Spatial Atlas: Spatial Intelligence Engine

THE CROWN JEWEL: Structured spatial scene graph for deterministic reasoning.

Instead of asking the LLM to hallucinate spatial relationships:
1. EXTRACT spatial entities and relationships from vision descriptions (LLM)
2. STORE in a queryable scene graph (structured data)
3. COMPUTE distances, containment, violations DETERMINISTICALLY (math)
4. FEED computed facts back to LLM for natural language generation

This separation yields:
- Correct distance measurements (not hallucinated)
- Accurate violation counts
- Consistent JSON structures for json_match evaluation
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from llm import LLMClient

logger = logging.getLogger("spatial-atlas.fieldwork.spatial")


@dataclass
class SpatialEntity:
    """A physical object/person in the scene."""

    id: str
    label: str  # "forklift", "worker", "shelf_unit"
    position: tuple[float, float] | None = None  # estimated (x, y) in meters
    attributes: dict[str, Any] = field(default_factory=dict)
    zone: str | None = None  # "loading_dock", "aisle_3"


@dataclass
class SpatialRelation:
    """A spatial relationship between entities."""

    subject: str  # entity id
    predicate: str  # "near", "blocking", "inside", "above", "left_of"
    object: str  # entity id or zone name
    distance: float | None = None  # estimated distance in meters


@dataclass
class SpatialScene:
    """Complete spatial scene graph with queryable operations."""

    entities: dict[str, SpatialEntity] = field(default_factory=dict)
    relations: list[SpatialRelation] = field(default_factory=list)
    zones: dict[str, dict[str, Any]] = field(default_factory=dict)
    safety_rules: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def add_entity(self, entity: SpatialEntity) -> None:
        self.entities[entity.id] = entity

    def add_relation(self, relation: SpatialRelation) -> None:
        self.relations.append(relation)

    def compute_distance(self, id_a: str, id_b: str) -> float | None:
        """Compute Euclidean distance between two entities."""
        a = self.entities.get(id_a)
        b = self.entities.get(id_b)
        if not a or not b or not a.position or not b.position:
            return None
        dx = a.position[0] - b.position[0]
        dy = a.position[1] - b.position[1]
        return math.sqrt(dx * dx + dy * dy)

    def compute_all_distances(self) -> None:
        """Compute distances for all relations that don't have one."""
        for rel in self.relations:
            if rel.distance is None:
                dist = self.compute_distance(rel.subject, rel.object)
                if dist is not None:
                    rel.distance = round(dist, 2)

    def query_near(self, entity_id: str, radius: float) -> list[SpatialEntity]:
        """Find entities within radius of given entity."""
        results = []
        for eid, entity in self.entities.items():
            if eid == entity_id:
                continue
            dist = self.compute_distance(entity_id, eid)
            if dist is not None and dist <= radius:
                results.append(entity)
        return results

    def check_constraints(self) -> list[str]:
        """Check safety rules and report violations."""
        self.violations = []
        for rule in self.safety_rules:
            rule_lower = rule.lower()

            # PPE checks
            if "ppe" in rule_lower or "safety vest" in rule_lower or "hard hat" in rule_lower:
                for entity in self.entities.values():
                    if entity.label.lower() in ("worker", "person", "employee"):
                        attrs = {k.lower(): v for k, v in entity.attributes.items()}
                        if not attrs.get("wearing_ppe", True):
                            self.violations.append(
                                f"{entity.label} ({entity.id}) missing PPE in {entity.zone}"
                            )
                        if "hard hat" in rule_lower and not attrs.get("hard_hat", True):
                            self.violations.append(f"{entity.label} ({entity.id}) missing hard hat")
                        if "safety vest" in rule_lower and not attrs.get("safety_vest", True):
                            self.violations.append(
                                f"{entity.label} ({entity.id}) missing safety vest"
                            )

            # Distance-based checks (e.g., "maintain 3m from forklifts")
            if "distance" in rule_lower or "meters" in rule_lower or "from" in rule_lower:
                self._check_distance_rule(rule)

        return self.violations

    def _check_distance_rule(self, rule: str) -> None:
        """Parse and check distance-based safety rules."""
        import re

        # Try to extract: "X must be Y meters from Z"
        dist_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meters?)", rule.lower())
        if not dist_match:
            return

        required_dist = float(dist_match.group(1))

        # Check all worker-to-hazard distances
        for rel in self.relations:
            if rel.distance is not None and rel.distance < required_dist:
                subj = self.entities.get(rel.subject)
                obj = self.entities.get(rel.object)
                if subj and obj:
                    subj_is_person = subj.label.lower() in ("worker", "person", "employee")
                    obj_is_hazard = obj.label.lower() in (
                        "forklift",
                        "machinery",
                        "crane",
                        "conveyor",
                        "vehicle",
                    )
                    if subj_is_person and obj_is_hazard:
                        self.violations.append(
                            f"{subj.label} ({subj.id}) too close to {obj.label} "
                            f"({obj.id}): {rel.distance:.1f}m < {required_dist}m required"
                        )

    def to_fact_sheet(self) -> str:
        """Convert scene to a computed fact sheet for LLM consumption."""
        facts = []

        if self.entities:
            facts.append("## Entities Detected")
            for entity in self.entities.values():
                pos_str = (
                    f" at ({entity.position[0]:.1f}, {entity.position[1]:.1f})"
                    if entity.position
                    else ""
                )
                zone_str = f" in {entity.zone}" if entity.zone else ""
                attrs_str = (
                    f" [{', '.join(f'{k}={v}' for k, v in entity.attributes.items())}]"
                    if entity.attributes
                    else ""
                )
                facts.append(f"- {entity.label} ({entity.id}){pos_str}{zone_str}{attrs_str}")

        if self.relations:
            facts.append("\n## Spatial Relationships")
            for rel in self.relations:
                if rel.distance is None:
                    dist_str = ""
                elif rel.predicate == "horizontal_surface_gap":
                    dist_str = f" (distance: {rel.distance:.6f}m)"
                else:
                    dist_str = f" (distance: {rel.distance:.1f}m)"
                facts.append(f"- {rel.subject} {rel.predicate} {rel.object}{dist_str}")

        if self.zones:
            facts.append("\n## Zones")
            for name, info in self.zones.items():
                zone_type = info.get("type", "unknown")
                facts.append(f"- {name}: {zone_type}")

        if self.violations:
            facts.append("\n## SAFETY VIOLATIONS DETECTED")
            for v in self.violations:
                facts.append(f"- VIOLATION: {v}")
        elif self.safety_rules:
            facts.append("\n## Safety Status: No violations detected")

        return "\n".join(facts) if facts else ""

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def violation_count(self) -> int:
        return len(self.violations)


class SpatialAnalyzer:
    """Build spatial scenes from visual descriptions using LLM extraction."""

    EXTRACTION_PROMPT = """Analyze the following content and extract spatial information for a scene graph.

Content:
{context}

Extract all physical entities, their spatial relationships, zones/areas, and applicable safety rules.

Return as JSON (be thorough: include every object, person, vehicle, and piece of equipment):
{{
  "entities": [
    {{
      "id": "e1",
      "label": "worker",
      "position_x": 3.0,
      "position_y": 7.0,
      "zone": "loading_dock",
      "attributes": {{"wearing_ppe": true, "hard_hat": true, "safety_vest": false, "activity": "loading"}}
    }}
  ],
  "relations": [
    {{
      "subject": "e1",
      "predicate": "near",
      "object": "e2",
      "distance_meters": 2.5
    }}
  ],
  "zones": [
    {{
      "name": "loading_dock",
      "type": "hazard_zone"
    }}
  ],
  "safety_rules": [
    "Workers must wear safety vests in hazard zones",
    "Maintain 3m distance from forklifts"
  ]
}}

If spatial positions cannot be determined, omit position_x/position_y but still include the entity.
If distances cannot be estimated, omit distance_meters but still include the relation."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def build_scene(self, query: str, file_contexts: list[str]) -> SpatialScene:
        """Extract spatial information and build a scene graph."""
        context = "\n\n".join(file_contexts)

        # Use LLM to extract structured spatial data
        prompt = self.EXTRACTION_PROMPT.format(context=context[:8000])

        try:
            result = await self.llm.generate(
                prompt,
                model_tier="strong",
                json_mode=True,
                max_tokens=4096,
            )
            data = json.loads(result)
        # Deliberately broad: an LLM call plus a json.loads, and any failure of either
        # degrades to an empty scene rather than taking the request down. Written as a
        # bare Exception because json.JSONDecodeError subclasses ValueError subclasses
        # Exception, so naming it alongside Exception caught nothing extra and only
        # implied a JSON-specific path that does not exist.
        except Exception as e:
            logger.warning(f"Spatial extraction failed, returning empty scene: {e}")
            return SpatialScene()

        # Build the scene graph
        scene = SpatialScene()

        # Add entities
        for e in data.get("entities", []):
            pos = None
            if e.get("position_x") is not None and e.get("position_y") is not None:
                pos = (float(e["position_x"]), float(e["position_y"]))
            entity = SpatialEntity(
                id=e.get("id", f"e{len(scene.entities)}"),
                label=e.get("label", "unknown"),
                position=pos,
                attributes=e.get("attributes", {}),
                zone=e.get("zone"),
            )
            scene.add_entity(entity)

        # Add relations
        for r in data.get("relations", []):
            rel = SpatialRelation(
                subject=r.get("subject", ""),
                predicate=r.get("predicate", "near"),
                object=r.get("object", ""),
                distance=r.get("distance_meters"),
            )
            scene.add_relation(rel)

        # Add zones
        for z in data.get("zones", []):
            scene.zones[z.get("name", "unknown")] = {
                "type": z.get("type", "general"),
            }

        # Add safety rules
        scene.safety_rules = data.get("safety_rules", [])

        # Compute derived facts deterministically
        scene.compute_all_distances()
        scene.check_constraints()

        logger.info(
            f"Scene built: {scene.entity_count} entities, "
            f"{len(scene.relations)} relations, "
            f"{scene.violation_count} violations"
        )
        return scene
