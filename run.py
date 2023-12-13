import subprocess
import matplotlib.pyplot as plt
import json

iter_range = 100
num_items = 20

for iter in range(0, iter_range):
  test_case = subprocess.run(f"./testgen --numitems = {num_items} --seedrand={iter}", capture_output=True, text=True)
  