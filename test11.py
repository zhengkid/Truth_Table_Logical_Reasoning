class Person:
    def __init__(
        self,
        name,
        is_student=None,
        attends_events=None,
        performs_in_talent_shows=None,
        chaperone_dances=None,
        is_active=None
    ):
        self.name = name
        self.is_student = is_student
        self.attends_events = attends_events
        self.performs_in_talent_shows = performs_in_talent_shows
        self.chaperone_dances = chaperone_dances
        self.is_active = is_active

def apply_premises(person: Person) -> bool:
    changed = False

    # Premise 1: People in this club who perform in school talent shows often attend and are very engaged with school events.
    if person.name == "Bonnie":
        if person.attends_events is None or person.attends_events is False:
            person.attends_events = True
            changed = True
        if person.performs_in_talent_shows is None:
            person.performs_in_talent_shows = True
            changed = True

    # Premise 2: People in this club either perform in school talent shows often or are inactive and disinterested community members.
    if person.name == "Bonnie":
        if person.is_active is not None and person.is_active is True:
            if person.attends_events is None or person.attends_events is False:
                person.attends_events = False
                person.is_active = False
                changed = True
            if person.performs_in_talent_shows is None:
                person.performs_in_talent_shows = False
                changed = True

    # Premise 3: People in this club who chaperone high school dances are not students who attend the school.
    if person.name == "Bonnie":
        if person.chaperone_dances is not None and person.chaperone_dances is True:
            if person.is_student is not None and person.is_student is True:
                person.is_student = False
                changed = True

    # Premise 4: All people in this club who are inactive and disinterested members of their community chaperone high school dances.
    if person.name == "Bonnie":
        if person.is_active is None or person.is_active is False:
            if person.chaperone_dances is None:
                person.chaperone_dances = True
                changed = True

    # Premise 5: All young children and teenagers in this club who wish to further their academic careers and educational opportunities are students who attend the school.
    # (No update is made here as it does not affect the conclusion directly.)
    return changed

def run_inference(person: Person):
    while apply_premises(person):
        pass

def check_conclusion(person: Person) -> str:
    run_inference(person)
    if person.performs_in_talent_shows is None:
        return "Uncertain"
    if person.performs_in_talent_shows is True:
        return "True"
    else:
        return "False"

def func():
    person = Person(
        name="Bonnie",
        is_student=None,
        attends_events=None,
        performs_in_talent_shows=None,
        chaperone_dances=None,
        is_active=None
    )
    return check_conclusion(person)

result = func()
print("Conclusion: Bonnie performs in school talent shows often?", result)
