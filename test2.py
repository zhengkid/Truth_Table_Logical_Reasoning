from z3 import Solver, DeclareSort, Function, ForAll, Implies, And, Or, Not, BoolSort, Const

# Declare sorts
Person = DeclareSort("Person")  # People in the club
Event = DeclareSort("Event")  # School events

# Define predicates
Is_In_Club = Function("Is_In_Club", Person, BoolSort())  # In(x, club)
Performs_In_Talent_Show = Function("Performs_In_Talent_Show", Person, BoolSort())  # PerformOften(x, talentShow)
Attends_Event = Function("Attends_Event", Person, Event, BoolSort())  # Attend(x, event)
Is_Engaged = Function("Is_Engaged", Person, Event, BoolSort())  # Engaged(x, event)
Chaperones_Event = Function("Chaperones_Event", Person, Event, BoolSort())  # Chaperone(x, event)
Is_Inactive = Function("Is_Inactive", Person, BoolSort())  # Inactive(x)
Is_Community_Member = Function("Is_Community_Member", Person, BoolSort())  # InactiveAndDisinterested(x)
Wishes_To_Further_Academic_Career = Function("Wishes_To_Further_Academic_Career", Person, BoolSort())  # YoungChildOrTeenager(x) ∧ WantsToFurther(x, academicCareer, educationalOpportunities)
Is_Student = Function("Is_Student", Person, BoolSort())  # Student(x)

# Declare constants
Bonnie = Const("Bonnie", Person)
x = Const("x", Person)
event = Const("event", Event)

# Define solver
solver = Solver()

# **Premises**
solver.add(ForAll([x], Implies(And(Is_In_Club(x), Performs_In_Talent_Show(x)), And(Attends_Event(x, event), Is_Engaged(x, event)))))  # ∀x (In(x, club) ∧ PerformOften(x, talentShow) → Attend(x, event) ∧ Engaged(x, event))
solver.add(ForAll([x], Implies(And(Is_In_Club(x), Or(Performs_In_Talent_Show(x), Is_Inactive(x))), True)))  # ∀x (In(x, club) ∧ (PerformOften(x, talentShow) ⊕ Inactive(x)))
solver.add(ForAll([x], Implies(And(Is_In_Club(x), Chaperones_Event(x, event)), Not(Is_Student(x)))))  # ∀x (In(x, club) ∧ Chaperone(x, event) → ¬Student(x))
solver.add(ForAll([x], Implies(And(Is_In_Club(x), Is_Inactive(x)), Chaperones_Event(x, event))))  # ∀x (In(x, club) ∧ Inactive(x) → Chaperone(x, event))
solver.add(ForAll([x], Implies(And(Is_In_Club(x), Wishes_To_Further_Academic_Career(x)), Is_Student(x))))  # ∀x (In(x, club) ∧ YoungChildOrTeenager(x) ∧ WantsToFurther(x, academicCareer, educationalOpportunities) → Student(x))

# **Bonnie's Conditions**
solver.add(Is_In_Club(Bonnie))  # Bonnie is in the club
solver.add(Or(And(Attends_Event(Bonnie, event), Is_Engaged(Bonnie, event), Is_Student(Bonnie)), Not(And(Attends_Event(Bonnie, event), Is_Engaged(Bonnie, event), Is_Student(Bonnie)))))  # Bonnie is either both attends and is very engaged with school events and is a student who attends the school or is not someone who both attends and is very engaged with school events and is not a student who attends the school

# **Conclusion to Check**
conclusion = Implies(Or(Chaperones_Event(Bonnie, event), Not(Chaperones_Event(Bonnie, event))), And(Wishes_To_Further_Academic_Career(Bonnie), Is_Inactive(Bonnie)))  # If Bonnie either chaperones high school dances or, if she does not, she performs in school talent shows often, then Bonnie is both a young child or teenager who wishes to further her academic career and educational opportunities and an inactive and disinterested member of the community.

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
    print("The conclusion is necessarily true: If Bonnie either chaperones high school dances or, if she does not, she performs in school talent shows often, then Bonnie is both a young child or teenager who wishes to further her academic career and educational opportunities and an inactive and disinterested member of the community. (True)")
elif result1 == sat and result2 == unsat:
    print("The conclusion is necessarily false: If Bonnie either chaperones high school dances or, if she does not, she performs in school talent shows often, then Bonnie is not both a young child or teenager who wishes to further her academic career and educational opportunities and an inactive and disinterested member of the community. (False)")
elif result1 == sat and result2 == sat:
    print("The conclusion is uncertain: It depends on additional unstated assumptions. (Uncertain)")
else:
    print("Unexpected result, possible logical error.")
