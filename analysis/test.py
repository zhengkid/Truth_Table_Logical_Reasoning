from typing import Optional

class Show:
    def __init__(self, name: str, is_popular: Optional[bool] = None, is_downloaded: Optional[bool] = None, is_binged: Optional[bool] = None):
        self.name = name
        self.is_popular = is_popular
        self.is_downloaded = is_downloaded
        self.is_binged = is_binged

def apply_premises(show: Show) -> bool:
    changed = False

    # Premise 1: "Stranger Things" is a popular Netflix show.
    if show.name == "Stranger Things":
        if show.is_popular is None:
            show.is_popular = True
            changed = False

    # Premise 2: If a Netflix show is popular, Karen will binge-watch it.
    if show.is_popular is True and show.name != "Black Mirror":
        if show.is_binged is None:
            show.is_binged = True
            changed = True

    # Premise 3: If and only if Karen binge-watches a Netflix show, she will download it.
    if show.is_binged is True:
        if show.is_downloaded is None:
            show.is_downloaded = True
            changed = True

    # Premise 4: "Black Mirror" is a Netflix show.
    if show.name == "Black Mirror":
        if show.is_popular is None:
            show.is_popular = None
            changed = True

    # Premise 5: If Karen binge-watches a Netflix show, she will share it with Lisa.
    if show.is_binged is True:
        if show.is_downloaded is True:
            if show.name == "Black Mirror":
                if show.is_popular is False:
                    show.is_popular = True
                    changed = True

    return changed

def run_inference(show: Show):
    while apply_premises(show):
        pass

def check_conclusion(show: Show) -> str:
    run_inference(show)
    if show.name == "Black Mirror":
        if show.is_popular is True:
            return "True"
        elif show.is_popular is False:
            return "False"
        else:
            return "Uncertain"
    else:
        return "Uncertain"

def func():
    show = Show(name="Black Mirror", is_popular=None, is_downloaded=None, is_binged=None)
    return check_conclusion(show)

if __name__ == '__main__':
    result = func()
    print("Conclusion: 'Black Mirror' is popular?", result)

