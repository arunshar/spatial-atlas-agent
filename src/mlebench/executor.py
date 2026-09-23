"""
Spatial Atlas: bounded code executor

Runs generated ML pipeline scripts in a subprocess with timeout.
Captures stdout/stderr for debugging and bounded repair.
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

logger = logging.getLogger("spatial-atlas.mlebench.executor")


class ExecutionOutputLimitError(RuntimeError):
    """Raised when generated code exceeds a captured output limit."""


class CodeExecutor:
    """Execute ML pipeline code safely in a subprocess."""

    def __init__(
        self,
        timeout: int = 600,
        *,
        max_code_bytes: int = 2 * 1024 * 1024,
        max_stream_bytes: int = 8 * 1024 * 1024,
        max_submission_bytes: int = 64 * 1024 * 1024,
        termination_grace_seconds: float = 0.2,
    ):
        self.timeout = timeout
        self.max_code_bytes = max_code_bytes
        self.max_stream_bytes = max_stream_bytes
        self.max_submission_bytes = max_submission_bytes
        self.termination_grace_seconds = termination_grace_seconds
        self.last_stdout: str = ""
        self.last_stderr: str = ""
        self.last_error: str | None = None

    async def execute(
        self,
        code: str,
        working_dir: Path,
        submission_path: Path | None = None,
    ) -> bytes | None:
        """
        Execute ML code in subprocess, return submission.csv bytes.

        Args:
            code: Complete Python script to execute
            working_dir: Directory to run in (contains data/)
            submission_path: Where to find submission.csv (default: working_dir/submission.csv)

        Returns:
            submission.csv bytes if produced, None if execution failed
        """
        encoded_code = code.encode("utf-8")
        if len(encoded_code) > self.max_code_bytes:
            self.last_error = (
                f"Generated code exceeds the {self.max_code_bytes}-byte execution limit"
            )
            return None

        script_path = working_dir / "pipeline.py"
        script_path.write_bytes(encoded_code)

        if submission_path is None:
            submission_path = working_dir / "submission.csv"
        submission_path.unlink(missing_ok=True)

        self.last_stdout = ""
        self.last_stderr = ""
        self.last_error = None

        logger.info(f"Executing pipeline.py in {working_dir} (timeout={self.timeout}s)")

        proc = None
        process_group_id = None
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_path),
                cwd=str(working_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._safe_env(working_dir),
                start_new_session=True,
            )
            process_group_id = proc.pid

            stdout, stderr = await asyncio.wait_for(
                self._communicate_bounded(proc), timeout=self.timeout
            )

            self.last_stdout = stdout.decode("utf-8", errors="replace")
            self.last_stderr = stderr.decode("utf-8", errors="replace")

            # Log output for debugging
            if self.last_stdout:
                logger.info(f"Pipeline stdout (last 500 chars): {self.last_stdout[-500:]}")
            if self.last_stderr:
                logger.warning(f"Pipeline stderr (last 500 chars): {self.last_stderr[-500:]}")

            if proc.returncode != 0:
                self.last_error = (
                    f"Script exited with code {proc.returncode}.\n"
                    f"Stderr:\n{self.last_stderr[-2000:]}"
                )
                logger.error(f"Pipeline failed: exit code {proc.returncode}")
                return None

            # The direct process can exit while a descendant continues writing the
            # output file. Stop the complete process group before inspecting it.
            await self._terminate_process_group(proc, process_group_id)
            process_group_id = None

            # Check for submission file
            if submission_path.exists():
                with submission_path.open("rb") as stream:
                    csv_bytes = stream.read(self.max_submission_bytes + 1)
                if len(csv_bytes) > self.max_submission_bytes:
                    self.last_error = (
                        f"Submission exceeds the {self.max_submission_bytes}-byte output limit"
                    )
                    logger.error(self.last_error)
                    return None
                logger.info(f"Submission produced: {len(csv_bytes)} bytes")
                return csv_bytes
            else:
                self.last_error = (
                    f"Script ran successfully but did not produce {submission_path.name}.\n"
                    f"Stdout:\n{self.last_stdout[-1000:]}"
                )
                logger.error("No submission.csv found after execution")
                return None

        except TimeoutError:
            self.last_error = f"Code execution timed out after {self.timeout}s"
            logger.error(self.last_error)
            return None
        except ExecutionOutputLimitError as exc:
            self.last_error = str(exc)
            logger.error(self.last_error)
            return None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.last_error = f"Execution error: {e}"
            logger.error(f"Pipeline execution exception: {e}")
            return None
        finally:
            await self._terminate_process_group(proc, process_group_id)

    async def _read_bounded(self, stream: asyncio.StreamReader, label: str) -> bytes:
        """Read one subprocess stream without allowing unbounded buffering."""
        payload = bytearray()
        while True:
            chunk = await stream.read(64 * 1024)
            if not chunk:
                return bytes(payload)
            payload.extend(chunk)
            if len(payload) > self.max_stream_bytes:
                raise ExecutionOutputLimitError(
                    f"Pipeline {label} exceeds the {self.max_stream_bytes}-byte capture limit"
                )

    async def _communicate_bounded(self, proc) -> tuple[bytes, bytes]:
        """Drain stdout and stderr concurrently with independent limits."""
        if proc.stdout is None or proc.stderr is None:
            raise RuntimeError("Generated-code subprocess streams are unavailable")
        stdout_task = asyncio.create_task(self._read_bounded(proc.stdout, "stdout"))
        stderr_task = asyncio.create_task(self._read_bounded(proc.stderr, "stderr"))
        tasks = (stdout_task, stderr_task)
        try:
            stdout, stderr = await asyncio.gather(*tasks)
            await proc.wait()
            return stdout, stderr
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    @staticmethod
    def _process_group_exists(process_group_id: int) -> bool:
        """Report whether the group is still ours to signal.

        EPERM counts as gone rather than present. The kernel recycles process-group ids, so
        once our descendants exit the id can be reissued to a group owned by another user,
        and killpg then reports EPERM instead of ESRCH. We can always signal our own
        children, so EPERM means the id is no longer ours.

        Treating it as present was wrong twice over. It raised PermissionError straight out
        of the termination path, which is how this surfaced: an intermittent failure under
        repeated runs, where pid reuse becomes likely. Worse, had the recycled group belonged
        to this same user, the caller would have gone on to SIGTERM and SIGKILL an unrelated
        process group.
        """
        try:
            os.killpg(process_group_id, 0)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    async def _terminate_process_group(self, proc, process_group_id: int | None) -> None:
        """Terminate the complete generated-code process group, including stragglers."""
        if proc is None or process_group_id is None:
            return

        if self._process_group_exists(process_group_id):
            try:
                os.killpg(process_group_id, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

            deadline = asyncio.get_running_loop().time() + self.termination_grace_seconds
            while (
                self._process_group_exists(process_group_id)
                and asyncio.get_running_loop().time() < deadline
            ):
                await asyncio.sleep(0.02)

            if self._process_group_exists(process_group_id):
                try:
                    os.killpg(process_group_id, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

        if proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()

    def _safe_env(self, working_dir: Path) -> dict[str, str]:
        """Build a minimal environment without parent-process credentials."""
        env = {
            key: os.environ[key]
            for key in ("PATH", "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE")
            if os.environ.get(key)
        }
        private_home = working_dir / ".runtime-home"
        private_tmp = working_dir / ".runtime-tmp"
        private_home.mkdir(mode=0o700, exist_ok=True)
        private_tmp.mkdir(mode=0o700, exist_ok=True)
        env.update(
            {
                "HOME": str(private_home),
                "TMPDIR": str(private_tmp),
                "PYTHONHASHSEED": "42",
                "PYTHONWARNINGS": "ignore",
            }
        )
        return env
