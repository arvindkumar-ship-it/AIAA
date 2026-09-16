"""
Central logging layer for AIAA.
Every module (calendar_integration, intervention, gating) imports log_task_event
from here so all real-run data lands in ONE table -> that table is your dataset
for retraining and your evidence for the impact-comparison report.

Local dev default: SQLite (aiaa.db file, zero setup).
Production: set DATABASE_URL to a Postgres URL and it switches automatically.

Setup (SQLite, default -- just this):
    pip install sqlalchemy python-dotenv

Setup (Postgres, optional, for production):
    pip install psycopg2-binary
    set DATABASE_URL=postgresql://user:password@localhost:5432/aiaa
"""

import os
import uuid
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, Float, DateTime, JSON
)
from sqlalchemy.orm import sessionmaker, declarative_base

# Defaults to a local SQLite file if DATABASE_URL isn't set -- no Postgres
# install required for local dev/testing. Set DATABASE_URL env var to a
# postgresql://... URL to use Postgres instead (no code change needed).
# Absolute path (relative to this file, not cwd) so every module that
# imports db.py from a different folder still hits the SAME database file.
_DEFAULT_SQLITE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aiaa.db")
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{_DEFAULT_SQLITE_PATH}")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
Session = sessionmaker(bind=engine)
Base = declarative_base()


class TaskLog(Base):
    __tablename__ = "task_logs"

    id = Column(Integer, primary_key=True)
    task_id = Column(String(64), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    task_type = Column(String(64))
    context_json = Column(JSON, nullable=False)

    rule_decision = Column(Boolean)
    rule_threshold = Column(Float)
    model_prob = Column(Float)
    final_decision = Column(Boolean)
    decision_source = Column(String(16))

    outcome_success = Column(Boolean)
    outcome_detail = Column(JSON)
    user_feedback = Column(String(16))

    execution_time_ms = Column(Integer)


def init_db():
    """Call once at startup. Creates the table if it doesn't exist."""
    Base.metadata.create_all(engine)


# Auto-create on import as a safety net -- so any module that imports
# log_task_event directly (without remembering to call init_db() first)
# doesn't hit "no such table" the way gating_logic.py just did.
init_db()


def log_task_event(
    task_type: str,
    context: dict,
    rule_decision: bool = None,
    rule_threshold: float = None,
    model_prob: float = None,
    final_decision: bool = None,
    decision_source: str = None,
    outcome_success: bool = None,
    outcome_detail: dict = None,
    user_feedback: str = None,
    execution_time_ms: int = None,
    task_id: str = None,
) -> str:
    """
    Call this from EVERY task run — calendar, bill pay, whatever.
    Returns the task_id so you can update outcome_success later
    once you know if the real action actually succeeded.
    """
    task_id = task_id or str(uuid.uuid4())
    session = Session()
    try:
        row = TaskLog(
            task_id=task_id,
            task_type=task_type,
            context_json=context,
            rule_decision=rule_decision,
            rule_threshold=rule_threshold,
            model_prob=model_prob,
            final_decision=final_decision,
            decision_source=decision_source,
            outcome_success=outcome_success,
            outcome_detail=outcome_detail,
            user_feedback=user_feedback,
            execution_time_ms=execution_time_ms,
        )
        session.add(row)
        session.commit()
        return task_id
    finally:
        session.close()


def update_outcome(task_id: str, outcome_success: bool, outcome_detail: dict = None):
    """Call this after you get a real confirmation (or failure) from Calendar/BBPS API."""
    session = Session()
    try:
        row = session.query(TaskLog).filter_by(task_id=task_id).first()
        if row:
            row.outcome_success = outcome_success
            row.outcome_detail = outcome_detail
            session.commit()
    finally:
        session.close()


def export_training_data(task_type: str = None):
    """
    Pulls every logged row as a list of dicts — this is what you feed
    into intervention/train_intervention.py once you have enough real rows.
    """
    session = Session()
    try:
        q = session.query(TaskLog)
        if task_type:
            q = q.filter_by(task_type=task_type)
        rows = q.all()
        return [
            {
                "task_id": r.task_id,
                "context": r.context_json,
                "rule_decision": r.rule_decision,
                "final_decision": r.final_decision,
                "outcome_success": r.outcome_success,
                "user_feedback": r.user_feedback,
            }
            for r in rows
        ]
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
    print("Table created / verified against", DATABASE_URL)