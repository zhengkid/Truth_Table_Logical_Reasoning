class Hulk:
    def __init__(self, angry):
        self.angry = angry

    @property
    def wakes_up(self):
        return self.angry

    @property
    def breaks_bridge(self):
        return self.wakes_up

class Thor:
    def __init__(self, happy):
        self.happy = happy

    @property
    def breaks_bridge(self):
        return self.happy

class PeterParker:
    def __init__(self, is_superhero=None):
        self.is_superhero = is_superhero

    @property
    def wears_uniform(self):
        return self.is_superhero

def apply_premises(thor, hulk, peter):
    updated = False
    if thor.happy and not hulk.angry:
        hulk.angry = True
        updated = True
    if hulk.breaks_bridge and peter.is_superhero is not True:
        peter.is_superhero = True
        updated = True
    return updated

def run_inference(thor, hulk, peter):
    while apply_premises(thor, hulk, peter):
        pass

def check_conclusion(thor, hulk, peter):
    run_inference(thor, hulk, peter)
    if not thor.happy:
        return True
    return peter.wears_uniform

def func():
    thor = Thor(happy=True)
    hulk = Hulk(angry=True)
    peter = PeterParker(is_superhero=False)
    return check_conclusion(thor, hulk, peter)

if __name__ == '__main__':
    result = func()
    print("Conclusion: If Thor is happy, does Peter Parker wear a uniform? Answer:", result)

