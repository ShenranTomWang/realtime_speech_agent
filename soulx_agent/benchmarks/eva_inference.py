"""Run EVA's actual conversation workers now; defer validation and scoring to eval."""

import json
from datetime import datetime


def install():
    from eva.models.results import ConversationResult, RunResult
    from eva.orchestrator.runner import BenchmarkRunner

    async def inference(
        self, *, pending_ids, all_output_ids, output_id_to_record, started_at, **kwargs
    ):
        await self.port_pool.initialize()
        completed = []
        failed = []
        for output_id in pending_ids:
            result_path = self.output_dir / "records" / output_id / "result.json"
            if result_path.exists():
                previous = ConversationResult.model_validate_json(result_path.read_text())
                if previous.completed:
                    completed.append(output_id)
                    continue
            result, audio_task = await self._run_conversation(
                output_id_to_record[output_id], output_id
            )
            if audio_task:
                await audio_task
            if not isinstance(result, ConversationResult):
                raise RuntimeError(f"EVA conversation worker failed: {output_id}") from result
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(result.model_dump_json(indent=2))
            (completed if result.completed else failed).append(output_id)
        summary = {
            "phase": "inference",
            "validation": "deferred_to_eval",
            "completed_record_ids": completed,
            "failed_record_ids": failed,
        }
        (self.output_dir / "inference_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        return RunResult(
            run_id=self.config.run_id,
            total_records=len(all_output_ids),
            successful_records=len(completed),
            failed_records=len(failed),
            duration_seconds=(datetime.now() - started_at).total_seconds(),
        )

    async def resume(self, run_dir, records, force_revalidation=False):
        return await self.run(records)

    BenchmarkRunner._run = inference
    BenchmarkRunner.validate_existing = resume
