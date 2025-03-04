from z3 import Solver, DeclareSort, Function, ForAll, Implies, And, Or, Not, BoolSort, Const

# Declare sorts
Person = DeclareSort("Person")  # Entities (Peter Parker, Hulk, Thor)
Object = DeclareSort("Object")  # Objects (bridge, uniform)

# Define predicates
Is_Superhero = Function("Is_Superhero", Person, BoolSort())  # Superhero(x)
Is_Civilian = Function("Is_Civilian", Person, BoolSort())  # Civilian(x)
Is_Destroyer = Function("Is_Destroyer", Person, BoolSort())  # Destroyer(x)
Is_God = Function("Is_God", Person, BoolSort())  # God(x)
Is_Angry = Function("Is_Angry", Person, BoolSort())  # Angry(x)
Is_Happy = Function("Is_Happy", Person, BoolSort())  # Happy(x)
Wakes_Up = Function("Wakes_Up", Person, BoolSort())  # WakesUp(x)
Breaks = Function("Breaks", Person, Object, BoolSort())  # Breaks(x, y)
Wears = Function("Wears", Person, Object, BoolSort())  # Wears(x, y)

# Declare constants
PeterParker = Const("PeterParker", Person)
TheHulk = Const("TheHulk", Person)
Thor = Const("Thor", Person)
Bridge = Const("Bridge", Object)
Uniform = Const("Uniform", Object)

# Define solver
solver = Solver()

# **Premises**
solver.add(Or(Is_Superhero(PeterParker), Is_Civilian(PeterParker)))  # Superhero(peterParker) ⊕ Civilian(peterParker)
solver.add(Not(And(Is_Superhero(PeterParker), Is_Civilian(PeterParker))))  # Ensuring XOR condition
solver.add(Is_Destroyer(TheHulk))  # Destroyer(theHulk)
solver.add(Implies(Is_Angry(TheHulk), Wakes_Up(TheHulk)))  # Angry(theHulk) → WakesUp(theHulk)
solver.add(Implies(Wakes_Up(TheHulk), Breaks(TheHulk, Bridge)))  # WakesUp(theHulk) → Breaks(theHulk, bridge)
solver.add(Is_God(Thor))  # God(thor)
solver.add(Implies(Is_Happy(Thor), Breaks(Thor, Bridge)))  # Happy(thor) → Breaks(thor, bridge)
solver.add(ForAll([x], Implies(Is_God(x), Not(Is_Destroyer(x)))))  # ∀x (God(x) → ¬Destroyer(x))
solver.add(Implies(Is_Superhero(PeterParker), Wears(PeterParker, Uniform)))  # Superhero(peter) → Wears(peter, uniform)
solver.add(ForAll([x], Implies(And(Is_Destroyer(x), Breaks(x, Bridge)), Not(Is_Civilian(PeterParker)))))  # ∀x ((Destroyer(x) ∧ Breaks(x,bridge)) → ¬Civilian(peter))
solver.add(Implies(Is_Happy(Thor), Is_Angry(TheHulk)))  # Happy(thor) → Angry(theHulk)

# **Conclusion to Check**
conclusion = Implies(Is_Happy(Thor), Wears(PeterParker, Uniform))  # Happy(thor) → Wears(peterParker, uniform)

# **Check validity**
solver.push()
solver.add(Not(conclusion))  # Negate the conclusion and check for contradiction
result1 = solver.check()
solver.pop()

solver.push()
solver.add(conclusion)
result2 = solver.check()
solver.pop()

# **Corrected Decision Logic**
from z3 import unsat, sat

if result1 == unsat:
    print("The conclusion is necessarily true: If Thor is happy, then Peter Parker wears a uniform. (True)")
    result = 'True'
elif result1 == sat and result2 == unsat:
    print("The conclusion is necessarily false: If Thor is happy, Peter Parker may not wear a uniform. (False)")
    result = 'False'
elif result1 == sat and result2 == sat:
    print("The conclusion is uncertain: It depends on additional unstated assumptions. (Uncertain)")
    result = 'Uncertain'
else:
    print("Unexpected result, possible logical error.")
    result = 'Unknown'
