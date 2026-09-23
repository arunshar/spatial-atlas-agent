"""
Spatial Atlas: Agent Unit Tests

Tests domain classification, message parsing, and formatting logic
(without requiring actual API keys or LLM calls).
"""

import json

from config import Config
from fieldwork.formatter import AnswerFormatter
from fieldwork.parser import GoalParser


class TestDomainClassification:
    """Test the domain classifier in Agent."""

    def _make_agent(self):
        """Create agent with mocked LLM to avoid API calls."""
        # Import here to test import chain
        from agent import Agent

        agent = Agent.__new__(Agent)
        agent.config = Config()
        agent.messages = []
        return agent

    def test_fieldwork_detection(self):
        agent = self._make_agent()
        text = "# Question\nHow many workers?\n# Input Data\nimage.jpg\n# Output Format\nnumber"
        assert agent._classify_domain(text, []) == "fieldwork"

    def test_mlebench_detection_by_file(self):
        agent = self._make_agent()
        file_parts = [("competition.tar.gz", "application/gzip", b"fake")]
        assert agent._classify_domain("some text", file_parts) == "mlebench"

    def test_mlebench_detection_by_keyword(self):
        agent = self._make_agent()
        text = "You are participating in a Kaggle competition. Train a model."
        assert agent._classify_domain(text, []) == "mlebench"

    def test_default_to_fieldwork(self):
        agent = self._make_agent()
        assert agent._classify_domain("random text", []) == "fieldwork"


class TestGoalParser:
    """Test FieldWorkArena goal string parsing."""

    def test_standard_goal(self):
        parser = GoalParser()
        goal = (
            "# Question\nHow many forklifts?\n# Input Data\n"
            "image1.jpg image2.jpg\n# Output Format\nnumber"
        )
        task = parser.parse(goal)
        assert "forklifts" in task.query
        assert task.output_format == "number"

    def test_missing_sections(self):
        parser = GoalParser()
        task = parser.parse("Just a plain question about safety")
        assert task.query == "Just a plain question about safety"
        assert task.output_format == ""


class TestAnswerFormatter:
    """Test output format matching."""

    def test_format_numeric(self):
        fmt = AnswerFormatter()
        assert fmt.format_answer("There are 5 workers", "number") == "5"
        assert fmt.format_answer("42", "integer") == "42"

    def test_format_boolean(self):
        fmt = AnswerFormatter()
        assert fmt.format_answer("Yes, it is", "yes/no") == "yes"
        assert fmt.format_answer("No violation found", "yes or no") == "no"

    def test_format_json(self):
        fmt = AnswerFormatter()
        result = fmt.format_answer('{"count": 3}', "json")
        parsed = json.loads(result)
        assert parsed["count"] == 3

    def test_format_json_extraction(self):
        fmt = AnswerFormatter()
        result = fmt.format_answer(
            'The answer is {"workers": 5, "violations": 2} based on my analysis.', "json"
        )
        parsed = json.loads(result)
        assert parsed["workers"] == 5

    def test_format_list(self):
        fmt = AnswerFormatter()
        result = fmt.format_answer("- item1\n- item2\n- item3", "list")
        assert "item1" in result
        assert "item2" in result

    def test_strip_markdown(self):
        fmt = AnswerFormatter()
        result = fmt.format_answer("**bold** and `code`", "text")
        assert result == "bold and code"

    def test_passthrough(self):
        fmt = AnswerFormatter()
        result = fmt.format_answer("Just a plain answer", "free text")
        assert result == "Just a plain answer"


class TestSpatialScene:
    """Test spatial scene graph computations."""

    def test_distance_computation(self):
        from fieldwork.spatial import SpatialEntity, SpatialRelation, SpatialScene

        scene = SpatialScene()
        scene.add_entity(SpatialEntity(id="w1", label="worker", position=(0.0, 0.0)))
        scene.add_entity(SpatialEntity(id="f1", label="forklift", position=(3.0, 4.0)))
        scene.add_relation(SpatialRelation(subject="w1", predicate="near", object="f1"))

        scene.compute_all_distances()
        assert scene.relations[0].distance == 5.0  # 3-4-5 triangle

    def test_query_near(self):
        from fieldwork.spatial import SpatialEntity, SpatialScene

        scene = SpatialScene()
        scene.add_entity(SpatialEntity(id="w1", label="worker", position=(0.0, 0.0)))
        scene.add_entity(SpatialEntity(id="f1", label="forklift", position=(2.0, 0.0)))
        scene.add_entity(SpatialEntity(id="f2", label="forklift", position=(10.0, 0.0)))

        nearby = scene.query_near("w1", 5.0)
        assert len(nearby) == 1
        assert nearby[0].id == "f1"

    def test_ppe_violation(self):
        from fieldwork.spatial import SpatialEntity, SpatialScene

        scene = SpatialScene()
        scene.add_entity(
            SpatialEntity(
                id="w1",
                label="worker",
                zone="loading_dock",
                attributes={"wearing_ppe": False},
            )
        )
        scene.safety_rules = ["Workers must wear PPE"]
        violations = scene.check_constraints()
        assert len(violations) >= 1
        assert "PPE" in violations[0] or "ppe" in violations[0].lower()

    def test_fact_sheet(self):
        from fieldwork.spatial import SpatialEntity, SpatialScene

        scene = SpatialScene()
        scene.add_entity(SpatialEntity(id="w1", label="worker", position=(1.0, 2.0), zone="dock"))
        facts = scene.to_fact_sheet()
        assert "worker" in facts
        assert "dock" in facts

    def test_fact_sheet_preserves_zero_distance(self):
        from fieldwork.spatial import SpatialEntity, SpatialRelation, SpatialScene

        scene = SpatialScene()
        scene.add_entity(SpatialEntity("a", "Region 0", (1.0, 1.0)))
        scene.add_entity(SpatialEntity("b", "Region 1", (1.0, 1.0)))
        scene.add_relation(SpatialRelation("a", "distance_to", "b", 0.0))

        assert "distance: 0.0m" in scene.to_fact_sheet()

    def test_empty_scene(self):
        from fieldwork.spatial import SpatialScene

        scene = SpatialScene()
        assert scene.entity_count == 0
        assert scene.violation_count == 0
        assert scene.to_fact_sheet() == ""


class TestCostTracker:
    """Test cost tracking utilities."""

    def test_tracker_init(self):
        from cost.tracker import CostTracker

        tracker = CostTracker()
        assert tracker.stats.total_tokens == 0
        assert tracker.has_budget()

    def test_budget_exceeded(self):
        from cost.tracker import CostTracker

        tracker = CostTracker(max_tokens=150_000)
        tracker.stats.prompt_tokens = 100_000
        tracker.stats.completion_tokens = 60_000
        tracker.stats.total_tokens = 160_000
        assert not tracker.has_budget()


class TestConfig:
    """Test configuration."""

    def test_default_config(self):
        config = Config()
        assert config.max_code_iterations == 3
        assert config.max_video_frames == 30
        assert "fast" in config.model_tiers
        assert "strong" in config.model_tiers
