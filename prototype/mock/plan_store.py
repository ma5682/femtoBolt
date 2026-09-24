"""Persist demonstration plans only; never accesses production outputs."""
import json
import os
from pathlib import Path


class MockPlanStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def save(self, record: dict) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / (record["id"] + ".json")
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def records(self) -> list[dict]:
        records = []
        if self.directory.exists():
            for path in self.directory.glob("mock-plan-*.json"):
                try:
                    item = json.loads(path.read_text(encoding="utf-8"))
                    if item.get("kind") == "mock_planning_preview":
                        records.append(item)
                except (OSError, ValueError, TypeError):
                    continue
        return sorted(records, key=lambda r: r.get("saved_at", ""), reverse=True)
