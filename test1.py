code_str = """
from typing import Optional

class Animal:
    def __init__(self, is_extinct: Optional[bool] = None):
        self.is_extinct = is_extinct

class HeckCattle(Animal):
    pass

class Aurochs(Animal):
    pass

def apply_premises(heck_cattle: HeckCattle, aurochs: Aurochs) -> bool:
    changed = False

    # Premise 1: Breeding back is a form of artificial selection by the deliberate selective breeding of domestic animals.
    # This premise does not provide information about the extinction status of aurochs.

    # Premise 2: Heck cattle were bred back in the 1920s to resemble the aurochs.
    # This premise establishes a relationship between heck cattle and aurochs, but does not state whether aurochs are extinct.

    # Premise 3: Heck cattle are animals.
    # This premise is already reflected in the class definition of HeckCattle.

    # Premise 4: Aurochs are animals.
    # This premise is already reflected in the class definition of Aurochs.

    # Premise 5: Some animals to be bred back resemble extinct animals.
    # This premise does not directly state whether aurochs are extinct.

    return changed

def run_inference(heck_cattle: HeckCattle, aurochs: Aurochs):
    while apply_premises(heck_cattle, aurochs):
        pass

def check_conclusion(heck_cattle: HeckCattle, aurochs: Aurochs) -> str:
    run_inference(heck_cattle, aurochs)
    if aurochs.is_extinct is not None:
        if aurochs.is_extinct is True:
            return "True"
        else:
            return "False"
    else:
        return "Uncertain"

def func():
    heck_cattle = HeckCattle()
    aurochs = Aurochs()
    return check_conclusion(heck_cattle, aurochs)

if __name__ == '__main__':
    result = func()
    print("Conclusion: Aurochs are extinct?", result)


"""

exec(code_str)
