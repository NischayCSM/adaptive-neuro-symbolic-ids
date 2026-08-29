import pytest
import tempfile
import os
from src.rules.knowledge_base import KnowledgeBase, RuleStatus, ConflictResolver

@pytest.fixture
def temp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.remove(path)

def test_kb_save_and_reload(temp_db_path):
    # Initialize KB and add a rule
    kb = KnowledgeBase(db_path=temp_db_path)
    
    rule_data = {
        "predicted_class": "DoS",
        "conditions": [{"feature": "feat_1", "operator": ">", "threshold": 0.5}],
        "precision": 0.95,
        "coverage": 0.80,
        "confidence": 0.88,
        "num_observations": 1500,
        "status": RuleStatus.ACTIVE
    }
    
    rule_id = kb.add_rule(rule_data)
    
    # Verify rule was added
    active_rules = kb.get_active_rules()
    assert len(active_rules) == 1
    assert active_rules[0]["rule_id"] == rule_id
    assert active_rules[0]["predicted_class"] == "DoS"
    assert active_rules[0]["conditions"][0]["feature"] == "feat_1"
    
    # Reload KB from the SAME path and verify persistence
    kb2 = KnowledgeBase(db_path=temp_db_path)
    
    reloaded_active_rules = kb2.get_active_rules()
    assert len(reloaded_active_rules) == 1
    
    rule2 = reloaded_active_rules[0]
    assert rule2["rule_id"] == rule_id
    assert rule2["predicted_class"] == "DoS"
    assert rule2["precision"] == 0.95
    assert rule2["num_observations"] == 1500
    assert rule2["status"] == RuleStatus.ACTIVE
    assert len(rule2["conditions"]) == 1
    assert rule2["conditions"][0]["operator"] == ">"
    
    # Update and Retire
    kb2.update_rule(rule_id, {"precision": 0.98})
    updated_rules = kb2.get_rules_for_class("DoS")
    assert updated_rules[0]["precision"] == 0.98
    
    kb2.retire_rule(rule_id)
    
    # Verify retired
    assert len(kb2.get_active_rules()) == 0
    
    summary = kb2.export_summary()
    assert summary["by_status"].get(RuleStatus.RETIRED) == 1
    assert summary["by_status"].get(RuleStatus.ACTIVE, 0) == 0
    assert summary["by_class"].get("DoS") == 1

def test_conflict_resolver_disjoint():
    rule_a = {"conditions": [{"feature": "f1", "operator": ">", "threshold": 10}]}
    rule_b = {"conditions": [{"feature": "f1", "operator": "<=", "threshold": 5}]}
    assert ConflictResolver.check_overlap(rule_a["conditions"], rule_b["conditions"]) == False

def test_conflict_resolver_overlap():
    rule_a = {"conditions": [{"feature": "f1", "operator": ">", "threshold": 5}]}
    rule_b = {"conditions": [{"feature": "f1", "operator": "<=", "threshold": 10}]}
    assert ConflictResolver.check_overlap(rule_a["conditions"], rule_b["conditions"]) == True

def test_conflict_resolver_resolve():
    rule_a = {
        "rule_id": "a",
        "conditions": [{"feature": "f1", "operator": ">", "threshold": 5}],
        "last_updated": "2023-01-01T00:00:00",
        "num_observations": 100,
        "precision": 0.90
    }
    rule_b = {
        "rule_id": "b",
        "conditions": [{"feature": "f1", "operator": "<=", "threshold": 10}],
        "last_updated": "2023-01-01T00:00:00",
        "num_observations": 50,  # lower obs -> weaker
        "precision": 0.80
    }
    # a has more obs and higher precision, b should be retired
    res = ConflictResolver.resolve(rule_a, rule_b)
    assert res["action"] == "RETIRE"
    assert res["target_rule_id"] == "b"
