"""Leave one condition (setting) out, with inner folds that leave one *training* condition out."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class InnerFold:
    validation_condition: str
    train_clip_ids: list[str]
    validation_clip_ids: list[str]


@dataclass(frozen=True)
class Fold:
    held_out_condition: str
    test_clip_ids: list[str]
    train_clip_ids: list[str]
    inner_folds: list[InnerFold]


def _group_by_condition(clips: Iterable[Mapping[str, str]]) -> dict[str, list[str]]:
    seen: set[str] = set()
    groups: dict[str, list[str]] = {}
    for clip in clips:
        clip_id, condition = clip.get("clip_id", ""), clip.get("condition_id", "")
        if not clip_id or not condition:
            raise ValueError("every clip needs a clip_id and a condition_id")
        if clip_id in seen:
            raise ValueError(f"duplicate clip_id {clip_id!r}")
        seen.add(clip_id)
        groups.setdefault(condition, []).append(clip_id)
    return {condition: sorted(ids) for condition, ids in sorted(groups.items())}


def _hold_out_each(groups: Mapping[str, list[str]]) -> list[tuple[str, list[str], list[str]]]:
    return [
        (held, groups[held], sorted(cid for cond, ids in groups.items() if cond != held for cid in ids))
        for held in groups
    ]


def leave_one_condition_out(clips: Iterable[Mapping[str, str]]) -> list[Fold]:
    groups = _group_by_condition(clips)
    if len(groups) < 2:
        raise ValueError("leave-one-condition-out needs at least two conditions")
    folds = []
    for held, test_ids, train_ids in _hold_out_each(groups):
        remaining = {cond: ids for cond, ids in groups.items() if cond != held}
        inner = (
            [InnerFold(val, train, val_ids) for val, val_ids, train in _hold_out_each(remaining)]
            if len(remaining) >= 2
            else []
        )
        folds.append(Fold(held, test_ids, train_ids, inner))
    return folds
