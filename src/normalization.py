from abc import ABC, abstractmethod
from typing import Any

class Normalization(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def add_value(self, value: Any) -> None:
        pass

    @abstractmethod
    def get_normalized(self) -> None:
        pass


class LastN(Normalization):
    def __init__(self) -> None:
        super().__init__()

    def add_value(self, value: Any) -> None:
        return super().add_value(value)

    def get_normalized(self) -> None:
        return super().get_normalized()


class PeriodicLastN(Normalization):
    def __init__(self) -> None:
        super().__init__()

    def add_value(self, value: Any) -> None:
        return super().add_value(value)

    def get_normalized(self) -> None:
        return super().get_normalized()
