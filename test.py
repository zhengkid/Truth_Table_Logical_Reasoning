
from typing import Optional

class ClubMember:
    def __init__(self,
                 performs_in_talent_shows: Optional[bool] = None,
                 attends_events: Optional[bool] = None,
                 engaged_with_events: Optional[bool] = None,
                 inactive_disinterested: Optional[bool] = None,
                 chaperones_dances: Optional[bool] = None,
                 student: Optional[bool] = None,
                 wishes_to_further_education: Optional[bool] = None):
        self.performs_in_talent_shows = performs_in_talent_shows
        self.attends_events = attends_events
        self.engaged_with_events = engaged_with_events
        self.inactive_disinterested = inactive_disinterested
        self.chaperones_dances = chaperones_dances
        self.student = student
        self.wishes_to_further_education = wishes_to_further_education

def apply_premises(member: ClubMember) -> bool:
    changed = False

    if member.performs_in_talent_shows is True and member.attends_events is not True:
        member.attends_events = True
        changed = True
    if member.performs_in_talent_shows is True and member.engaged_with_events is not True:
        member.engaged_with_events = True
        changed = True

    if member.performs_in_talent_shows is None and member.inactive_disinterested is not None:
        member.performs_in_talent_shows = member.inactive_disinterested
        changed = True

    if member.chaperones_dances is True and member.student is not False:
        member.student = False
        changed = True

    if member.inactive_disinterested is True and member.chaperones_dances is not True:
        member.chaperones_dances = True
        changed = True

    if member.wishes_to_further_education is True and member.student is not True:
        member.student = True
        changed = True

    if hasattr(member, 'name') and member.name == "Bonnie":
        if member.attends_events is not None and member.engaged_with_events is not None and member.student is not None:
            pass
        else:
            changed = True

    return changed

def run_inference(member: ClubMember):
    while apply_premises(member):
        print()
        pass

def check_conclusion(member: ClubMember) -> str:
    run_inference(member)
    return "True" if member.performs_in_talent_shows is True else "False"

def func():
    member = ClubMember()
    member.name = "Bonnie"
    return check_conclusion(member)

result = func()
print("Conclusion: Bonnie performs in school talent shows often?", result)


