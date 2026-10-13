"""In-memory cache of raw retrieved source data for ONE session (e.g. one conversation).

Clients that receive a SessionCache store what they downloaded and reuse it on later calls
in the same session. Nothing is written to disk and nothing survives the process.
Only successful retrievals are stored: a failed request raises before anything is cached.
Entries are raw source data (not final answers), keyed by namespace + key, for example
("ontime_form", ("Alaska", 2026, 4)) or ("t100_passengers", "ANC").
"""

from collections import Counter

_MISSING = object()


class SessionCache:
    def __init__(self):
        self._data = {}
        self.hits = Counter()      # per namespace
        self.misses = Counter()

    def get(self, namespace, key, default=None):
        value = self._data.get((namespace, key), _MISSING)
        if value is _MISSING:
            self.misses[namespace] += 1
            return default
        self.hits[namespace] += 1
        return value

    def contains(self, namespace, key):
        return (namespace, key) in self._data

    def put(self, namespace, key, value):
        self._data[(namespace, key)] = value

    def clear(self):
        self._data.clear()
        self.hits.clear()
        self.misses.clear()

    def stats(self):
        namespaces = Counter(namespace for namespace, _ in self._data)
        return {"entries": dict(namespaces), "hits": dict(self.hits), "misses": dict(self.misses)}
