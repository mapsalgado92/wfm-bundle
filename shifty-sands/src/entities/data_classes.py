from dataclasses import dataclass
from typing import List, Optional

# CLASSES THAT SHOULD BE IMORTED FROM A DB


@dataclass
class Shift:
    id: str  # unique id of the shift
    start_time: float  # between 0.0 and 24.0
    duration: float  # number of hours of shift duration
    task: str  # skill id shift provides coverage for


@dataclass
class Skill:
    id: str  # unique code defining skill
    task: str
    queue: str


@dataclass
class Queue:
    id: str
    name: str


@dataclass
class Task:
    id: str


@dataclass
class Agent:
    email: str
    queue: str
    active: bool


@dataclass
class Pattern:
    id: str
    mon: bool
    tue: bool
    wed: bool
    thu: bool
    fri: bool
    sat: bool
    sun: bool
