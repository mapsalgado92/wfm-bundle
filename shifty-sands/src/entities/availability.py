from typing import List


class DailyAvailability:
    def __init__(self, skill: str, is_off: bool = False, is_none: bool = True) -> None:
        self.skill = skill
        self.is_off = is_off
        self.is_none = is_none


class AgentAvailability:
    def __init__(
        self, avail_list: List[str], off_skill: str = "off", none_skill: str = ""
    ) -> None:
        self.availability = [
            DailyAvailability(
                skill=avail,
                is_off=True if avail == "off_skill" else False,
                is_none=True if avail == "none_skill" else False,
            )
            for avail in avail_list
        ]
        self.off_skill = off_skill
        self.none_skill = none_skill
        self.days = len(avail_list)


# class ConditionalAvailability:
# pass
