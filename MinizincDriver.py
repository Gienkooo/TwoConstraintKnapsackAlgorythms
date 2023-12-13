import minizinc
import json
import sys

def process_json_data(json_data):
    model = minizinc.Model("knapsack.mzn")
    data = minizinc.Instance(json_data)
    solver = minizinc.Solver.lookup("gecode")
    instance = minizinc.Instance(solver, model, data=data)
    result = instance.solve()

    print(result)

if __name__ == "__main__":
    try:
        # Read JSON data from stdin
        stdin_data = sys.stdin.read()

        # Load JSON data
        json_data = json.loads(stdin_data)

        # Process JSON data
        process_json_data(json_data)

    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)