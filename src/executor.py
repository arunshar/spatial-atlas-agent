"""
Spatial Atlas: A2A AgentExecutor

Routes incoming A2A requests to per-context Agent instances.
It started from the executor in the RDI Foundation's AgentBeats agent template.
"""

import logging
import uuid

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InvalidRequestError,
    TaskState,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError

from agent import Agent, AgentTaskError

logger = logging.getLogger("spatial-atlas.executor")


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class Executor(AgentExecutor):
    def __init__(self, agent_factory=Agent):
        self._agent_factory = agent_factory

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(
                error=InvalidRequestError(
                    message=f"Task {task.id} already processed (state: {task.status.state})"
                )
            )

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        updater = TaskUpdater(event_queue, task.id, context_id)
        await updater.start_work()

        try:
            agent = self._agent_factory()
            await agent.run(msg, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception as e:
            if isinstance(e, AgentTaskError):
                public_message = str(e)
                reference = e.reference
            else:
                reference = uuid.uuid4().hex[:12]
                public_message = (
                    f"Spatial Atlas could not complete this task. Reference: {reference}"
                )
            logger.error(
                "Task failed [reference=%s, exception_type=%s]",
                reference,
                type(e).__name__,
            )
            await updater.failed(
                new_agent_text_message(
                    public_message,
                    context_id=context_id,
                    task_id=task.id,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
