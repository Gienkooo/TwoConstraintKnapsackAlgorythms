import minizinc
import json
import sys
import numpy as np

def runMinizinc(json_data):   
    model = minizinc.Model("TwoConstraintKnapsack.mzn")
    #ddata = minizinc.Instance(json_data)
    solver = minizinc.Solver.lookup("gecode")
    instance = minizinc.Instance(solver, model)
    instance["maxsize"] = int(json_data["maxsize"])
    instance["maxweight"] = int(json_data["maxweight"])
    instance["n"] = int(json_data["n"])
    instance["sizes"] = np.array(json_data["sizes"])
    instance["weights"] = np.array(json_data["weights"])
    instance["values"] = np.array(json_data["values"])
    result = instance.solve()
    print(result.objective)

if __name__ == "__main__":
    try:
        stdin_data = sys.stdin.read()
        json_data = json.loads(stdin_data)
        runMinizinc(json_data)

    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)