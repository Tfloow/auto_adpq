# conftest.py
import pytest

def pytest_collection_modifyitems(items):
    # 1. Find items with the marker
    first_items = []
    other_items = []

    for item in items:
        if item.get_closest_marker("run_first"):
            first_items.append(item)
        else:
            other_items.append(item)

    # 2. Overwrite the items list with the new order
    items[:] = first_items + other_items