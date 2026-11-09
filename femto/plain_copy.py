"""Fast deep copy of JSON-shaped records (dict/list/tuple of plain scalars).

Values of any other type are handed to ``copy.deepcopy``. Unlike deepcopy no
memo is kept, so an object referenced twice is copied twice; route records are
trees built from JSON values, where that distinction is invisible.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import repeat
from operator import is_
from typing import Any

_ATOMS = (str, int, float, bool, type(None))


def plain_integers(values):
    """Scan every element using identity, without slices, type sets or equality."""
    return all(map(is_, map(type, values), repeat(int)))


def clone(value: Any) -> Any:
    kind = type(value)
    if kind is dict:
        return {key: (item if type(item) in _ATOMS else clone(item)) for key, item in value.items()}
    if kind is list:
        return [item if type(item) in _ATOMS else clone(item) for item in value]
    if kind in _ATOMS:
        return value
    if kind is tuple:
        return tuple(clone(item) for item in value)
    return deepcopy(value)
