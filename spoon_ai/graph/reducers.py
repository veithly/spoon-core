"""
Reducers and validators for the graph package.
"""
from typing import Any, Dict, List, Set
from datetime import datetime


def add_messages(existing: List[Any], new: List[Any]) -> List[Any]:
    if existing is None:
        existing = []
    if new is None:
        return existing
    return existing + new


def merge_dicts(existing: Dict, new: Dict) -> Dict:
    if existing is None:
        return new or {}
    if new is None:
        return existing
    result = existing.copy()
    for k, v in new.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def append_history(existing: List, new: Dict) -> List:
    if existing is None:
        existing = []
    if new is None:
        return existing
    entry = {"timestamp": datetime.now().isoformat(), **new}
    return existing + [entry]


def union_sets(existing: Set, new: Set) -> Set:
    if existing is None:
        existing = set()
    if new is None:
        return existing
    return existing | new


def validate_range(min_val: float, max_val: float):
    def validator(value: float) -> float:
        if not isinstance(value, (int, float)):
            raise ValueError(f"Value must be numeric, got {type(value)}")
        if not min_val <= value <= max_val:
            raise ValueError(f"Value {value} not in range [{min_val}, {max_val}]")
        return float(value)
    return validator


def validate_enum(allowed_values: List[Any]):
    def validator(value: Any) -> Any:
        if value not in allowed_values:
            raise ValueError(f"Value {value} not in allowed values: {allowed_values}")
        return value
    return validator

