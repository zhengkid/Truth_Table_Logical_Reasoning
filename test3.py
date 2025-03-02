from z3 import Solver, DeclareSort, Function, ForAll, Implies, And, Or, Not, BoolSort, Const

# Declare sorts
Person = DeclareSort("Person")  # Employees (James)
Location = DeclareSort("Location")  # Locations (Company Building, Home)

# Define predicates
Is_Employee = Function("Is_Employee", Person, BoolSort())  # Employee(x)
Schedules_Meeting = Function("Schedules_Meeting", Person, BoolSort())  # ScheduleMeeting(x, customer)
In_Company_Building = Function("In_Company_Building", Person, BoolSort())  # InCompanyBuilding(x)
Lunch_At_Home = Function("Lunch_At_Home", Person, BoolSort())  # LunchAtHome(x)
Working_Remotely = Function("Working_Remotely", Person, BoolSort())  # WorkingFromHome(x)
In_Other_Country = Function("In_Other_Country", Person, BoolSort())  # InOtherCountry(x)
Is_Manager = Function("Is_Manager", Person, BoolSort())  # Manager(x)
Appears_In_Company = Function("Appears_In_Company", Person, BoolSort())  # Appears(x, company)

# Declare constants
James = Const("James", Person)
Company_Building = Const("Company_Building", Location)
Home = Const("Home", Location)
x = Const("x", Person)

# Define solver
solver = Solver()

# **Premises**
solver.add(ForAll([x], Implies(And(Is_Employee(x), Schedules_Meeting(x)), In_Company_Building(x))))  # ∀x (Employee(x) ∧ ScheduleMeeting(x, customer) → InCompanyBuilding(x))
solver.add(ForAll([x], Implies(In_Company_Building(x), And(Is_Employee(x), Schedules_Meeting(x)))))  # ∀x (InCompanyBuilding(x) → Employee(x) ∧ ScheduleMeeting(x, customer))
solver.add(ForAll([x], Or(In_Company_Building(x), Lunch_At_Home(x))))  # ∀x (InCompanyBuilding(x) ⊕ LunchAtHome(x))
solver.add(ForAll([x], Implies(Lunch_At_Home(x), Working_Remotely(x))))  # ∀x (LunchAtHome(x) → WorkingFromHome(x))
solver.add(ForAll([x], Implies(In_Other_Country(x), Working_Remotely(x))))  # ∀x (InOtherCountry(x) → WorkingFromHome(x))
solver.add(ForAll([x], Not(And(Is_Employee(x), Working_Remotely(x)))))  # ∀x (Employee(x) → ¬WorkingFromHome(x))
solver.add(ForAll([x], Implies(Is_Manager(x), Not(Working_Remotely(x)))))  # ∀x (Manager(x) → ¬WorkingFromHome(x))
solver.add(ForAll([x], Implies(Is_Employee(x), And(Implies(Appears_In_Company(x), Is_Manager(x)), 
                Implies(Is_Manager(x), Appears_In_Company(x))))))  # ∀x (Employee(x) → (Appears(x, company) ⊃ Manager(x)))

# **James' Conditions**
solver.add(Is_Employee(James))  # James is an employee
solver.add(Not(Appears_In_Company(James)))  # James does not appear in the company

# **Conclusion to Check**
conclusion = Not(In_Company_Building(James))  # James does not have lunch in the company.

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
    print("The conclusion is necessarily true: James does not have lunch in the company. (True)")
    result = 'True'
elif result1 == sat and result2 == unsat:
    print("The conclusion is necessarily false: James has lunch in the company. (False)")
    result = 'False'
elif result1 == sat and result2 == sat:
    print("The conclusion is uncertain: It depends on additional unstated assumptions. (Uncertain)")
    result = 'Uncertain'
else:
    print("Unexpected result, possible logical error.")
    result = 'Unknown'
