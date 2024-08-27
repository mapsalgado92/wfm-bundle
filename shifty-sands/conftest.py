from copy import deepcopy
from tests.objects.object_fetcher import fetch_from_json

import pytest


class TestObjects:
    def __init__(self):
        # Dictionary to store test objects
        self._objects = {}

    def add_object(self, name, obj):
        """Adds an object to the TestObjects instance.

        Args:
          name: A unique identifier for the object.
          obj: The object to be stored.
        """
        self._objects[name] = obj

    def get_object(self, name):
        """Retrieves an object from the TestObjects instance.

        Args:
          name: The name of the object to retrieve.

        Returns:
          A deep copy of the requested object.
        """
        if name not in self._objects:
            raise KeyError(f"Object with name '{name}' not found")
        return deepcopy(self._objects[name])


@pytest.fixture(scope="session")
def test_objects() -> TestObjects:
    objects = TestObjects()
    objects.add_object("test_values", fetch_from_json("test_values"))
    return objects
