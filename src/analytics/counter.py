"""Bidirectional vehicle counter using virtual line-crossings."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from src.utils.logger import get_logger

log = get_logger("counter")


def _side_of_line(px, py, p1, p2):
    return (p2[0] - p1[0]) * (py - p1[1]) - (p2[1] - p1[1]) * (px - p1[0])


@dataclass
class LineCounter:
    id: str
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    in_count: int = 0
    out_count: int = 0
    per_class: Dict[int, Dict[str, int]] = field(default_factory=dict)
    _prev_side: Dict[int, float] = field(default_factory=dict)
    _counted: set = field(default_factory=set)

    def update(self, tracks: Sequence[dict]):
        for tr in tracks:
            tid = int(tr["id"])
            cx, cy = tr["centroid"]
            cls = int(tr.get("cls", 0))
            side = _side_of_line(cx, cy, self.p1, self.p2)
            prev = self._prev_side.get(tid)
            self._prev_side[tid] = side
            if prev is None:
                continue
            if prev * side < 0 and tid not in self._counted:
                if side > 0:
                    self.in_count += 1
                    self.per_class.setdefault(cls, {"in": 0, "out": 0})["in"] += 1
                else:
                    self.out_count += 1
                    self.per_class.setdefault(cls, {"in": 0, "out": 0})["out"] += 1
                self._counted.add(tid)
                log.debug("Line {} crossed by track {} ({})", self.id, tid, "IN" if side > 0 else "OUT")

    def summary(self) -> dict:
        return {"in": self.in_count, "out": self.out_count, "per_class": self.per_class}


class MultiLineCounter:
    def __init__(self, lines: List[dict]):
        self.lines = [LineCounter(id=l["id"], p1=tuple(l["points"][0]), p2=tuple(l["points"][1])) for l in lines]

    def update(self, tracks: Sequence[dict]):
        for l in self.lines:
            l.update(tracks)

    def summary(self) -> Dict[str, dict]:
        return {l.id: l.summary() for l in self.lines}
