from abc import ABC, abstractmethod
from typing import Any

class Generator(ABC):
    @abstractmethod
    def generate()->Any:
        pass