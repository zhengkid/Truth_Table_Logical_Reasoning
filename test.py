from z3 import Solver, Bool, And, Not, Implies

# Define propositional variables
Attended_Westminster = Bool("Attended_Westminster")
Attended_Edinburgh = Bool("Attended_Edinburgh")
Westminster_UK = Bool("Westminster_UK")
Edinburgh_UK = Bool("Edinburgh_UK")
Supported_Portland_Whigs = Bool("Supported_Portland_Whigs")
Not_Sat_In_Parliament = Bool("Not_Sat_In_Parliament")

# Initialize solver
solver = Solver()

# Add premises
solver.add(Attended_Westminster)  # William attended Westminster
solver.add(Attended_Edinburgh)  # William attended Edinburgh
solver.add(Edinburgh_UK)  # Edinburgh is in the UK
solver.add(Supported_Portland_Whigs)  # William supported Portland Whigs
solver.add(Implies(Supported_Portland_Whigs, Not_Sat_In_Parliament))  # Supporters of Portland Whigs did not sit in Parliament

# Define the conclusion
conclusion = And(Attended_Westminster, Attended_Edinburgh, Westminster_UK, Edinburgh_UK)

# Check if the conclusion is necessarily true
solver.push()
solver.add(Not(conclusion))
result1 = str(solver.check())
solver.pop()

solver.push()
solver.add(conclusion)
result2 = str(solver.check())
solver.pop()

# Interpret results
if result1 == "unsat":
    print("The conclusion is necessarily true.")
    result = "True"
elif result1 == "sat" and result2 == "unsat":
    print("The conclusion is necessarily false.")
    result = "False"
elif result1 == "sat" and result2 == "sat":
    print("The conclusion is uncertain.")
    result = "Uncertain"
else:
    print("Unexpected result, possible logical error.")
    result = "Error"

