"""
FastAPI REST Endpoints for Agent Orchestrator.

Endpoints:
- POST /tasks: Create and execute task
- GET /tasks/{task_id}: Get task status
- GET /users/{user_id}/style: Get user style
- GET /metrics: Get agent metrics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
from agent_orchestrator import AgentOrchestrator, Task
import structlog

logger = structlog.get_logger()

app = FastAPI(title="Autonomous Intervention-Aware Agent", version="1.0.0")

# Initialize agent
agent = AgentOrchestrator()


class TaskCreate(BaseModel):
    task_type: str
    user_id: str
    input_data: Dict[str, Any]


class TaskResponse(BaseModel):
    task_id: str
    status: str
    output_data: Optional[Dict[str, Any]]
    intervention_required: bool
    execution_time: float


@app.post("/tasks", response_model=TaskResponse)
async def create_task(task_data: TaskCreate, background_tasks: BackgroundTasks):
    """
    Create and execute a new task.
    """
    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        task_type=task_data.task_type,
        user_id=task_data.user_id,
        input_data=task_data.input_data,
    )
    
    # Execute task in background
    result = agent.execute_task(task)
    
    logger.info("Task created", task_id=task_id, status=result["status"])
    
    return TaskResponse(
        task_id=task_id,
        status=result["status"],
        output_data=result["output_data"],
        intervention_required=result["intervention_required"],
        execution_time=result["execution_time"],
    )


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """
    Get task status.
    """
    # In production, load from database
    # For now, return dummy data
    return {
        "task_id": task_id,
        "status": "completed",
        "output_data": {"completed_actions": ["navigate", "fill_form", "submit"]},
    }


@app.get("/users/{user_id}/style")
async def get_user_style(user_id: str):
    """
    Get user's collaboration style.
    """
    return agent.get_user_style(user_id)


@app.get("/metrics")
async def get_metrics():
    """
    Get agent performance metrics.
    """
    return agent.get_metrics()


@app.on_event("startup")
async def startup_event():
    logger.info("Agent API started")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Agent API shutting down")
