"""Session-bound delegation, inspection, steering, and cancellation tools."""

import asyncio
import uuid

from ..contracts import Tool


def build_tools(coordinator):
    async def submit(args):
        task_id = uuid.uuid4().hex
        coordinator.tasks[task_id] = {"status": "running", "request": args["request"]}
        coordinator.record("task.accepted", task_id=task_id, request=args["request"])
        coordinator.tasks[task_id]["agent"] = coordinator.harness(
            "backend", coordinator.backend_prompt, coordinator.environment_tools
        )
        coordinator.tasks[task_id]["future"] = asyncio.create_task(coordinator._work(task_id))
        return {"task_id": task_id, "status": "running"}

    async def status(args):
        return {
            task_id: {key: value for key, value in item.items() if key not in {"future", "agent"}}
            for task_id, item in coordinator.tasks.items()
        }

    async def check_status(args):
        task_id = args["task_id"]
        item = coordinator.tasks.get(task_id)
        if item is None:
            return {"error": "Unknown task", "task_id": task_id}
        return {
            "task_id": task_id,
            **{key: value for key, value in item.items() if key not in {"future", "agent"}},
        }

    async def cancel(args):
        task_id = args["task_id"]
        item = coordinator.tasks[task_id]
        if item["status"] == "running":
            item["status"] = "cancelled"
            item["future"].cancel()
            coordinator.record("task.cancelled", task_id=task_id)
        return {"status": item["status"]}

    async def steer(args):
        task_id = args["task_id"]
        item = coordinator.tasks.get(task_id)
        if item is None:
            return {"error": "Unknown task", "task_id": task_id}
        if item["status"] != "running":
            return {"error": "Only running tasks can be steered", "status": item["status"]}
        instruction = args["instruction"].strip()
        if not instruction:
            return {"error": "Steering instruction must not be empty"}
        accepted = await item["agent"].steer(instruction)
        if not accepted:
            return {"error": "Backend response has already finished", "task_id": task_id}
        item.setdefault("steering", []).append(instruction)
        coordinator.record("task.steered", task_id=task_id, instruction=instruction)
        return {"task_id": task_id, "status": item["status"], "steered": True}

    return [
        Tool(
            "submit_task",
            "Delegate a multi-step request and return immediately.",
            {
                "type": "object",
                "properties": {"request": {"type": "string"}},
                "required": ["request"],
            },
            submit,
        ),
        Tool(
            "task_status",
            "Inspect accepted tasks and their results.",
            {"type": "object", "properties": {}},
            status,
        ),
        Tool(
            "check_task_status",
            "Inspect one accepted task by ID, including its status and available result.",
            {
                "type": "object",
                "properties": {"task_id": {"type": "string", "minLength": 1}},
                "required": ["task_id"],
                "additionalProperties": False,
            },
            check_status,
        ),
        Tool(
            "steer_task",
            "Send updated instructions to a running backend task, retaining its context. "
            "Interrupts the current response; completed side effects cannot be undone.",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "instruction": {"type": "string", "minLength": 1},
                },
                "required": ["task_id", "instruction"],
            },
            steer,
        ),
        Tool(
            "cancel_task",
            "Explicitly cancel a task at the user's request. Completed side effects cannot be undone.",
            {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
            cancel,
        ),
    ]
