"""Share renderer progress with the local workbench."""
from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time


ROOT = Path(__file__).resolve().parent
DIRECTORY = Path("runtime/workbench")


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


class ProductionProgress:
    def __init__(self, manifest, output, *, engine=None, delivery=None, fresh=False):
        key = hashlib.sha256(os.fsencode(output.resolve())).hexdigest()[:16]
        self.path = ROOT / DIRECTORY / (key + ".json")
        self.enabled = True
        manifest = str(manifest.resolve())
        previous = read_json(self.path, {})
        self.data = previous if not fresh and previous.get("manifest") == manifest else {}
        if (fresh and previous.get("manifest") == manifest and previous.get("phase") == "planned"
                and previous.get("pid") == os.getpid()):
            self.data["events"] = previous.get("events", [])
        self.data.update(manifest=manifest, output=str(output.resolve()), pid=os.getpid(),
                         started=time.time())
        if engine:
            self.data["engine"] = engine
        if delivery:
            self.data["delivery"] = str(delivery.resolve())

    def update(self, phase, **values):
        now = time.time()
        if phase != self.data.get("phase"):
            self.data.setdefault("events", []).append({"time": now, "phase": phase})
        self.data.update(phase=phase, updated=now, **values)
        if not self.enabled:
            return
        tmp = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as out:
                tmp = Path(out.name)
                json.dump(self.data, out, ensure_ascii=False)
            tmp.replace(self.path)
        except OSError as exc:
            self.enabled = False
            logging.warning("Cannot save workbench progress: %s", exc)
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    def select(self, shot, receipt):
        self.data.setdefault("receipts", {})[shot] = str(receipt.resolve())
        self.update("sampling", current_shot=shot)


@contextmanager
def track(manifest, output, active, complete, **options):
    progress = ProductionProgress(manifest, output, **options)
    progress.update(active, error=None)
    try:
        yield progress
    except KeyboardInterrupt:
        progress.update("interrupted")
        raise
    except Exception as exc:
        progress.update("failed", error=str(exc).strip()[:600] or type(exc).__name__)
        raise
    else:
        progress.update(complete)
