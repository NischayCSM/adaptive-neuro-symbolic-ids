"""
KnowledgeBase module for managing neuro-symbolic IDS rules.

Storage Backend Choice: SQLite
Justification: SQLite was chosen over JSON because the KnowledgeBase needs to 
support querying rules by specific attributes (e.g., `get_active_rules`, 
`get_rules_for_class`). As the rule base grows (especially with continual 
learning and automated rule induction), loading a giant JSON file into memory 
and iterating through it becomes inefficient. SQLite provides lightweight, 
serverless ACID-compliant persistence with native querying capabilities that 
scale much better for our use-case without adding heavy infrastructure dependencies.
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
import contextlib

class RuleStatus:
    CANDIDATE = "CANDIDATE"
    VALIDATED = "VALIDATED"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    UPDATED = "UPDATED"
    RETIRED = "RETIRED"

class KnowledgeBase:
    def __init__(self, db_path: str = "data/processed/knowledge_base.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        with contextlib.closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rules (
                    rule_id TEXT PRIMARY KEY,
                    predicted_class TEXT,
                    conditions TEXT,
                    precision REAL,
                    coverage REAL,
                    confidence REAL,
                    created_at TEXT,
                    last_updated TEXT,
                    num_observations INTEGER,
                    status TEXT
                )
            ''')
            conn.commit()

    def add_rule(self, rule_data: dict) -> str:
        """
        Add a new rule to the knowledge base.
        rule_data expects: predicted_class, conditions, precision, coverage, confidence, num_observations, status
        rule_id is auto-generated if not provided.
        """
        rule_id = rule_data.get("rule_id", str(uuid.uuid4()))
        now = datetime.now(timezone.utc).isoformat()
        
        with contextlib.closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO rules (
                    rule_id, predicted_class, conditions, precision, coverage, 
                    confidence, created_at, last_updated, num_observations, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule_id,
                rule_data.get("predicted_class"),
                json.dumps(rule_data.get("conditions", [])),
                rule_data.get("precision", 0.0),
                rule_data.get("coverage", 0.0),
                rule_data.get("confidence", 0.0),
                now,
                now,
                rule_data.get("num_observations", 0),
                rule_data.get("status", RuleStatus.CANDIDATE)
            ))
            conn.commit()
            
        return rule_id

    def update_rule(self, rule_id: str, updates: dict):
        """
        Update specific fields of an existing rule.
        """
        if not updates:
            return
            
        now = datetime.now(timezone.utc).isoformat()
        updates["last_updated"] = now
        
        set_clauses = []
        values = []
        
        for k, v in updates.items():
            if k == "conditions":
                v = json.dumps(v)
            set_clauses.append(f"{k} = ?")
            values.append(v)
            
        values.append(rule_id)
        
        with contextlib.closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                UPDATE rules 
                SET {", ".join(set_clauses)}
                WHERE rule_id = ?
            ''', tuple(values))
            conn.commit()

    def retire_rule(self, rule_id: str):
        """
        Convenience method to set a rule's status to RETIRED.
        """
        self.update_rule(rule_id, {"status": RuleStatus.RETIRED})

    def _row_to_dict(self, row):
        return {
            "rule_id": row[0],
            "predicted_class": row[1],
            "conditions": json.loads(row[2]),
            "precision": row[3],
            "coverage": row[4],
            "confidence": row[5],
            "created_at": row[6],
            "last_updated": row[7],
            "num_observations": row[8],
            "status": row[9]
        }

    def get_active_rules(self) -> list:
        """
        Retrieve all rules with status = ACTIVE.
        """
        with contextlib.closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM rules WHERE status = ?', (RuleStatus.ACTIVE,))
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    def get_rules_for_class(self, predicted_class: str) -> list:
        """
        Retrieve all rules for a specific attack type, regardless of status.
        """
        with contextlib.closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM rules WHERE predicted_class = ?', (predicted_class,))
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    def export_summary(self) -> dict:
        """
        Export a summary of rule counts grouped by status and attack class.
        Returns:
            {
                "by_status": {"ACTIVE": 5, "RETIRED": 2, ...},
                "by_class": {"DoS": 4, "Exploits": 3, ...}
            }
        """
        summary = {
            "by_status": defaultdict(int),
            "by_class": defaultdict(int)
        }
        
        with contextlib.closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT status, COUNT(*) FROM rules GROUP BY status')
            for row in cursor.fetchall():
                summary["by_status"][row[0]] = row[1]
                
            cursor.execute('SELECT predicted_class, COUNT(*) FROM rules GROUP BY predicted_class')
            for row in cursor.fetchall():
                summary["by_class"][row[0]] = row[1]
                
        # Convert defaultdicts back to regular dicts for cleaner output/JSON serialization
        return {
            "by_status": dict(summary["by_status"]),
            "by_class": dict(summary["by_class"])
        }

class ConflictResolver:
    """Resolves conflicts between overlapping rules in the KnowledgeBase."""
    
    @staticmethod
    def check_overlap(conds_a: list, conds_b: list) -> bool:
        """
        Checks if two sets of rule conditions overlap in the feature space.
        If they are completely disjoint on ANY feature, they do not overlap.
        """
        intervals_a = {}
        intervals_b = {}
        
        # Parse intervals for A
        for c in conds_a:
            f = c["feature"]
            op = c["operator"]
            th = c["threshold"]
            if f not in intervals_a: intervals_a[f] = [-float('inf'), float('inf')]
            if op == '>': intervals_a[f][0] = max(intervals_a[f][0], th)
            elif op == '<=': intervals_a[f][1] = min(intervals_a[f][1], th)
            
        # Parse intervals for B
        for c in conds_b:
            f = c["feature"]
            op = c["operator"]
            th = c["threshold"]
            if f not in intervals_b: intervals_b[f] = [-float('inf'), float('inf')]
            if op == '>': intervals_b[f][0] = max(intervals_b[f][0], th)
            elif op == '<=': intervals_b[f][1] = min(intervals_b[f][1], th)
            
        # Check overlap
        for feature in set(intervals_a.keys()).intersection(set(intervals_b.keys())):
            min_a, max_a = intervals_a[feature]
            min_b, max_b = intervals_b[feature]
            
            # Disjoint check
            if max_a <= min_b or max_b <= min_a:
                return False # Disjoint on this feature = overall disjoint
                
        return True
        
    @staticmethod
    def resolve(rule_a: dict, rule_b: dict) -> dict:
        """
        Resolves a conflict. Returns a dict describing the action:
        {'action': 'COEXIST'}
        {'action': 'RETIRE', 'target_rule_id': id}
        {'action': 'NARROW', 'target_rule_id': id, 'new_conditions': [...]}
        """
        if not ConflictResolver.check_overlap(rule_a["conditions"], rule_b["conditions"]):
            return {"action": "COEXIST"}
            
        # Score rules: last_updated -> num_observations -> precision
        def score(r):
            # Parse last_updated to timestamp
            try:
                dt = datetime.fromisoformat(r["last_updated"]).timestamp()
            except:
                dt = 0
            return (dt, r["num_observations"], r["precision"])
            
        score_a = score(rule_a)
        score_b = score(rule_b)
        
        winner, loser = (rule_a, rule_b) if score_a > score_b else (rule_b, rule_a)
        
        # In a full implementation, we'd attempt to narrow the loser away from the winner.
        # For simplicity in this milestone, if there is a severe conflict, we just RETIRE the weaker rule.
        return {"action": "RETIRE", "target_rule_id": loser["rule_id"]}
