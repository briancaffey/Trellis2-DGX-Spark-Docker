"""Tiny in-memory store for generated results, so individual files can be
downloaded *unzipped* (one HTTP request per file) — handy for testing.

Results are held in RAM with a TTL and a total-size cap; oldest entries are
evicted first. This is deliberately simple (single-process, no persistence): it
exists so `response_mode="files"` can hand back per-file URLs without forcing the
client to deal with a zip. Restarting the server clears everything.
"""
import threading
import time
import uuid
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from . import config


# (filename, media_type, bytes)
FileTuple = Tuple[str, str, bytes]


class _Entry:
    __slots__ = ("created", "metadata", "files", "nbytes")

    def __init__(self, metadata: dict, files: List[FileTuple]):
        self.created = time.time()
        self.metadata = metadata
        self.files: Dict[str, Tuple[str, bytes]] = {
            name: (media_type, data) for name, media_type, data in files
        }
        self.nbytes = sum(len(data) for _n, _m, data in files)


class ResultsStore:
    def __init__(self, ttl_sec: int, max_total_bytes: int):
        self.ttl_sec = ttl_sec
        self.max_total_bytes = max_total_bytes
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()
        self._total = 0

    def _evict_locked(self):
        now = time.time()
        # Expire by age.
        for jid in [j for j, e in self._entries.items() if now - e.created > self.ttl_sec]:
            self._total -= self._entries.pop(jid).nbytes
        # Then by total size (oldest first).
        while self._total > self.max_total_bytes and self._entries:
            _jid, e = self._entries.popitem(last=False)
            self._total -= e.nbytes

    def put(self, metadata: dict, files: List[FileTuple]) -> str:
        entry = _Entry(metadata, files)
        job_id = uuid.uuid4().hex
        with self._lock:
            self._entries[job_id] = entry
            self._total += entry.nbytes
            self._evict_locked()
        return job_id

    def get(self, job_id: str) -> Optional[_Entry]:
        with self._lock:
            self._evict_locked()
            return self._entries.get(job_id)

    def get_file(self, job_id: str, filename: str) -> Optional[Tuple[str, bytes]]:
        entry = self.get(job_id)
        if entry is None:
            return None
        return entry.files.get(filename)


store = ResultsStore(
    ttl_sec=int(getattr(config, "RESULTS_TTL_SEC", 3600)),
    max_total_bytes=int(getattr(config, "RESULTS_MAX_BYTES", 2 * 1024 ** 3)),
)
