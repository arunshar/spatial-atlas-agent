"""
Spatial Atlas: MLE-Bench domain handler

Orchestrates the complete ML competition pipeline:
1. Extract competition data from tar.gz
2. Analyze competition description
3. Generate ML pipeline code
4. Execute code to produce submission.csv
5. Repair on failure (bounded retry with error context)
6. Return submission CSV bytes

Receives: instructions text + competition.tar.gz from green agent
Returns: (csv_bytes, summary_text)
"""

import base64
import binascii
import io
import logging
import os
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

import pandas as pd
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message

from config import Config
from llm import LLMClient
from mlebench.analyzer import CompetitionAnalyzer
from mlebench.codegen import MLCodeGenerator
from mlebench.executor import CodeExecutor

logger = logging.getLogger("spatial-atlas.mlebench")

MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 50_000
MAX_ARCHIVE_MEMBER_BYTES = 4 * 1024 * 1024 * 1024
MAX_ARCHIVE_EXPANDED_BYTES = 8 * 1024 * 1024 * 1024


class PipelineExecutionError(RuntimeError):
    """Raised when an MLE pipeline cannot produce a real submission."""


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# Matches lines like 'VALIDATION_SCORE: 0.8341' anywhere in stdout.
# Tolerates trailing text on the same line so the model can print a
# parenthetical comment after the number.
_VALIDATION_SCORE_RE = re.compile(r"VALIDATION_SCORE:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def _parse_validation_score(stdout: str) -> float | None:
    """
    Extract the last VALIDATION_SCORE emitted by the pipeline.

    Taking the LAST match (not first) lets a refined pipeline print a
    pre-training score followed by the real post-training score without
    us grabbing the wrong one.
    """
    if not stdout:
        return None
    matches = _VALIDATION_SCORE_RE.findall(stdout)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _score_is_better(new: float, old: float, metric_direction: str) -> bool:
    """
    Compare scores given the direction of the competition metric.

    metric_direction is a free-form string from CompetitionAnalysis. We
    treat it as 'maximize' unless it obviously says otherwise, because
    most Kaggle metrics in MLE-Bench are maximize (AUC, accuracy, F1).
    """
    direction = (metric_direction or "").lower()
    minimize_keywords = ("min", "lower", "loss", "error", "rmse", "mae", "mse")
    minimize = any(kw in direction for kw in minimize_keywords)
    return (new < old) if minimize else (new > old)


class MLEBenchHandler:
    """Handle MLE-Bench competition tasks end-to-end."""

    def __init__(self, config: Config, llm: LLMClient):
        self.config = config
        self.llm = llm
        self.analyzer = CompetitionAnalyzer(llm)
        self.codegen = MLCodeGenerator(llm)
        self.executor = CodeExecutor(timeout=config.code_execution_timeout)
        self._active_work_dirs: set[Path] = set()

    async def handle(
        self,
        text: str,
        file_parts: list[tuple[str, str, str | bytes]],
        updater: TaskUpdater,
    ) -> tuple[bytes, str]:
        """Run MLE code only when an operator explicitly enables the trusted path."""
        if not _enabled("ATLAS_ENABLE_MLEBENCH_CODE_EXECUTION") or not _enabled(
            "ATLAS_TRUSTED_ISOLATED_WORKER"
        ):
            raise PermissionError(
                "MLE-Bench code execution is disabled. An isolated worker must set both "
                "ATLAS_ENABLE_MLEBENCH_CODE_EXECUTION=true and "
                "ATLAS_TRUSTED_ISOLATED_WORKER=true."
            )
        existing = set(self._active_work_dirs)
        try:
            return await self._handle(text, file_parts, updater)
        finally:
            created = self._active_work_dirs - existing
            for work_dir in created:
                shutil.rmtree(work_dir, ignore_errors=True)
                self._active_work_dirs.discard(work_dir)

    async def _handle(
        self,
        text: str,
        file_parts: list[tuple[str, str, str | bytes]],
        updater: TaskUpdater,
    ) -> tuple[bytes, str]:
        """
        Process an MLE-Bench competition end-to-end.

        Args:
            text: Instructions text from green agent
            file_parts: List of (name, mime_type, data), should include competition.tar.gz
            updater: A2A task updater for progress reporting

        Returns:
            (csv_bytes, summary_text) tuple
        """
        # 1. Extract competition tar.gz to temp directory
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Extracting competition data..."),
        )

        work_dir = self._extract_competition(file_parts)
        data_dir = self._find_data_dir(work_dir)
        logger.info(f"Competition data extracted to {work_dir}, data at {data_dir}")

        # 2. Read competition description
        description = self._read_description(data_dir)
        file_listing = self._list_data_files(data_dir)
        data_preview = self._preview_data(data_dir)

        logger.info(f"Description: {len(description)} chars, files: {file_listing[:200]}")

        # 3. Analyze competition
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Analyzing competition..."),
        )

        analysis = await self.analyzer.analyze(
            description=description,
            file_listing=file_listing,
            data_preview=data_preview,
        )

        logger.info(
            f"Analysis: type={analysis.task_type}, metric={analysis.metric}, "
            f"strategy={analysis.strategy}"
        )

        # 4. Generate ML pipeline code
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Generating {analysis.strategy} ML pipeline..."),
        )

        submission_path = str(work_dir / "submission.csv")
        code = await self.codegen.generate(
            description=description,
            data_dir=str(data_dir),
            file_listing=file_listing,
            data_preview=data_preview,
            analysis=analysis,
            submission_path=submission_path,
        )

        # 5. Execute code with bounded repair retries
        csv_bytes = None
        for attempt in range(self.config.max_code_iterations):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Running ML pipeline (attempt {attempt + 1}/{self.config.max_code_iterations})..."
                ),
            )

            csv_bytes = await self.executor.execute(
                code=code,
                working_dir=work_dir,
                submission_path=Path(submission_path),
            )

            if csv_bytes is not None:
                logger.info(f"Pipeline succeeded on attempt {attempt + 1}")
                break

            # Repair: fix the code based on the error
            if attempt < self.config.max_code_iterations - 1:
                logger.info(f"Pipeline failed on attempt {attempt + 1}, repairing...")
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"Pipeline failed, fixing code (attempt {attempt + 2})..."
                    ),
                )

                code = await self.codegen.fix(
                    code=code,
                    error=self.executor.last_error or "Unknown error",
                    stdout=self.executor.last_stdout,
                    description=description,
                    file_listing=file_listing,
                )

        if csv_bytes is None:
            if not _enabled("ATLAS_ALLOW_DUMMY_SUBMISSION"):
                raise PipelineExecutionError(
                    "All generated pipeline attempts failed; no real submission was produced"
                )
            logger.warning("All attempts failed; emitting an explicitly enabled dummy submission")
            csv_bytes = self._generate_dummy_submission(data_dir, analysis)
            refinement_note = "dummy fallback explicitly enabled after pipeline failure"
        else:
            # 6. Score-driven refinement loop.
            #
            # The baseline pipeline already works; now ask the strong model
            # to propose targeted improvements. Each iteration re-runs the
            # full pipeline, parses VALIDATION_SCORE from stdout, and keeps
            # the better submission. Bail out on wall-time budget.
            csv_bytes, refinement_note = await self._refine_until_best(
                updater=updater,
                initial_code=code,
                initial_csv=csv_bytes,
                work_dir=work_dir,
                submission_path=Path(submission_path),
                description=description,
                file_listing=file_listing,
                analysis=analysis,
            )

        summary = (
            f"Competition: {analysis.task_type} ({analysis.metric})\n"
            f"Strategy: {analysis.strategy}\n"
            f"Submission: {len(csv_bytes)} bytes\n"
            f"Refinement: {refinement_note}"
        )
        return csv_bytes, summary

    async def _refine_until_best(
        self,
        *,
        updater: TaskUpdater,
        initial_code: str,
        initial_csv: bytes,
        work_dir: Path,
        submission_path: Path,
        description: str,
        file_listing: str,
        analysis,
    ) -> tuple[bytes, str]:
        """
        Run the score-driven refinement loop.

        Returns (best_csv_bytes, human_readable_note). The caller does not
        need to know how many iterations ran or whether any of them
        improved; the note is for the summary only.
        """
        max_iters = self.config.max_refinement_iterations
        wall_budget = self.config.refinement_wall_time_seconds

        if max_iters <= 0:
            return initial_csv, "disabled (max_refinement_iterations=0)"

        initial_score = _parse_validation_score(self.executor.last_stdout)
        if initial_score is None:
            logger.info(
                "Refinement skipped: no VALIDATION_SCORE found in pipeline stdout. "
                "Either the pipeline did not print one or parsing failed."
            )
            return initial_csv, "skipped (no VALIDATION_SCORE printed)"

        best_code = initial_code
        best_csv = initial_csv
        best_score = initial_score
        start = time.monotonic()
        improvements = 0

        logger.info(
            f"Refinement loop: baseline score={initial_score}, "
            f"max_iters={max_iters}, wall_budget={wall_budget}s"
        )

        for i in range(max_iters):
            if time.monotonic() - start > wall_budget:
                logger.info("Refinement wall-time budget exhausted; stopping")
                break

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Refining pipeline (iteration {i + 1}/{max_iters}, "
                    f"best score={best_score:.4f})..."
                ),
            )

            try:
                refined_code = await self.codegen.refine(
                    code=best_code,
                    current_score=best_score,
                    metric=analysis.metric,
                    metric_direction=analysis.metric_direction,
                    description=description,
                    file_listing=file_listing,
                )
            except Exception as e:
                logger.warning(f"Refinement codegen failed on iter {i + 1}: {e}")
                continue

            # Re-run with the refined script. On error we keep the previous
            # best and fall through; we do NOT call fix() here because fix
            # exists for first-pass errors, not for refinement regressions.
            csv_bytes = await self.executor.execute(
                code=refined_code,
                working_dir=work_dir,
                submission_path=submission_path,
            )
            if csv_bytes is None:
                logger.info(
                    f"Refined pipeline on iter {i + 1} failed to run; keeping previous best"
                )
                continue

            new_score = _parse_validation_score(self.executor.last_stdout)
            if new_score is None:
                logger.info(
                    f"Refined pipeline on iter {i + 1} ran but printed no "
                    "VALIDATION_SCORE; treating as non-improvement"
                )
                continue

            if _score_is_better(new_score, best_score, analysis.metric_direction):
                logger.info(
                    f"Refinement iter {i + 1}: improved {best_score:.4f} -> {new_score:.4f}"
                )
                best_code = refined_code
                best_csv = csv_bytes
                best_score = new_score
                improvements += 1
            else:
                logger.info(
                    f"Refinement iter {i + 1}: no improvement "
                    f"({new_score:.4f} vs {best_score:.4f}); keeping previous"
                )

        note = (
            f"baseline={initial_score:.4f}, best={best_score:.4f}, "
            f"improvements={improvements}/{max_iters}"
        )
        return best_csv, note

    def _extract_competition(self, file_parts: list[tuple[str, str, str | bytes]]) -> Path:
        """Extract competition.tar.gz to a temporary directory."""
        tar_data = None
        for name, mime, data in file_parts:
            if name and ("tar" in name or "gz" in name):
                tar_data = data
                break
            if mime and ("tar" in mime or "gzip" in mime):
                tar_data = data
                break

        if tar_data is None:
            # Use the first file attachment as tar
            if file_parts:
                _, _, tar_data = file_parts[0]
            else:
                raise ValueError("No competition data file received")

        # Decode if base64.
        if isinstance(tar_data, str):
            if tar_data.startswith("data:"):
                tar_data = tar_data.split(",", 1)[1]
            max_base64_chars = ((MAX_ARCHIVE_BYTES + 2) // 3) * 4 + 4
            if len(tar_data) > max_base64_chars:
                raise ValueError("Encoded competition archive exceeds the upload limit")
            try:
                tar_data = base64.b64decode(tar_data, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("Competition archive is not valid base64") from exc
        if not isinstance(tar_data, bytes):
            raise TypeError("Competition archive must be bytes or base64 text")
        if len(tar_data) > MAX_ARCHIVE_BYTES:
            raise ValueError("Decoded competition archive exceeds the upload limit")

        # Extract to temp directory
        work_dir = Path(tempfile.mkdtemp(prefix="atlas_mle_"))
        self._active_work_dirs.add(work_dir)
        try:
            try:
                with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:gz") as tar:
                    self._validate_and_extract(tar, work_dir)
            except tarfile.ReadError:
                # Try uncompressed tar.
                with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:") as tar:
                    self._validate_and_extract(tar, work_dir)
        except BaseException:
            shutil.rmtree(work_dir, ignore_errors=True)
            self._active_work_dirs.discard(work_dir)
            raise

        logger.info(f"Extracted competition to {work_dir}")
        return work_dir

    def _validate_and_extract(self, archive: tarfile.TarFile, work_dir: Path) -> None:
        """Reject unsafe or oversized members before writing archive contents."""
        members = archive.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError("Competition archive contains too many members")

        total_bytes = 0
        destinations: set[str] = set()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe competition archive path: {member.name!r}")
            normalized = str(path)
            if normalized in destinations:
                raise ValueError(f"Duplicate competition archive path: {member.name!r}")
            destinations.add(normalized)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported competition archive member: {member.name!r}")
            if member.size > MAX_ARCHIVE_MEMBER_BYTES:
                raise ValueError(f"Competition archive member is too large: {member.name!r}")
            total_bytes += member.size
            if total_bytes > MAX_ARCHIVE_EXPANDED_BYTES:
                raise ValueError("Competition archive expands beyond the configured limit")

        archive.extractall(work_dir, members=members, filter="data")

    def _find_data_dir(self, work_dir: Path) -> Path:
        """Find the data directory within extracted competition."""
        # MLE-Bench structure: home/data/
        candidates = [
            work_dir / "home" / "data",
            work_dir / "data",
            work_dir,
        ]
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.iterdir()):
                return candidate
        return work_dir

    def _read_description(self, data_dir: Path) -> str:
        """Read competition description."""
        for name in ["description.md", "README.md", "description.txt"]:
            desc_path = data_dir / name
            if desc_path.exists():
                return desc_path.read_text(errors="replace")

        # Look one level up
        parent = data_dir.parent
        for name in ["description.md", "README.md"]:
            desc_path = parent / name
            if desc_path.exists():
                return desc_path.read_text(errors="replace")

        return "[No description file found]"

    def _list_data_files(self, data_dir: Path) -> str:
        """List all data files with sizes."""
        lines = []
        for f in sorted(data_dir.rglob("*")):
            if f.is_file():
                rel = f.relative_to(data_dir)
                size = f.stat().st_size
                if size > 1_000_000:
                    size_str = f"{size / 1_000_000:.1f}MB"
                elif size > 1_000:
                    size_str = f"{size / 1_000:.1f}KB"
                else:
                    size_str = f"{size}B"
                lines.append(f"  - {rel} ({size_str})")
        return "\n".join(lines) if lines else "  [No files found]"

    def _preview_data(self, data_dir: Path, max_rows: int = 5) -> str:
        """Preview CSV files in the data directory."""
        previews = []
        csv_files = sorted(data_dir.glob("*.csv"))[:3]  # first 3 CSVs

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, nrows=max_rows)
                previews.append(
                    f"### {csv_path.name}\n"
                    f"Shape: {df.shape}\n"
                    f"Columns: {list(df.columns)}\n"
                    f"Dtypes:\n{df.dtypes.to_string()}\n"
                    f"Head:\n{df.to_string()}\n"
                )
            except Exception as e:
                previews.append(f"### {csv_path.name}\nError reading: {e}\n")

        return "\n".join(previews) if previews else "[No CSV files to preview]"

    def _generate_dummy_submission(self, data_dir: Path, analysis) -> bytes:
        """Generate a minimal valid submission as last resort."""
        # Try to read test.csv and create a dummy submission
        test_path = data_dir / "test.csv"
        if not test_path.exists():
            for candidate in data_dir.glob("test*.csv"):
                test_path = candidate
                break

        try:
            test_df = pd.read_csv(test_path)
            submission = pd.DataFrame()

            # Try to find ID column
            for col in test_df.columns:
                if "id" in col.lower():
                    submission[col] = test_df[col]
                    break

            # Add target column with dummy value
            if analysis.target_column:
                submission[analysis.target_column] = 0
            else:
                submission["target"] = 0

            buf = io.BytesIO()
            submission.to_csv(buf, index=False)
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Dummy submission failed: {e}")
            return b"id,target\n0,0\n"
