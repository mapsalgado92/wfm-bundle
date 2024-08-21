from dataclasses import dataclass
from typing import List, Optional

# CLASSES THAT SHOULD BE IMORTED FROM A DB


@dataclass
class Shift:
    id: str  # unique id of the shift
    start_time: float  # between 0.0 and 24.0
    duration: float  # number of hours of shift duration
    skill: str  # skill the shift provides coverage for


@dataclass
class Skill:
    code: str  # unique code defining skill


@dataclass
class Team:
    id: str


@dataclass
class Agent:
    employee_id: str
    email: str
    team: str
    skills: List[str]
