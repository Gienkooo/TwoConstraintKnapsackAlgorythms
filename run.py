import subprocess
import matplotlib.pyplot as plt
import numpy as np
import json
import time

num_tests = 100
num_items_list = [15, 20]
max_range = 100

def testGenerateCommand(num_items, minweight, maxweight, minsize, maxsize, seedrand = 0):
  command = ["./build/testgen", 
            f"--numitems={num_items}", 
            f"--minweight={minweight}", 
            f"--maxweight={maxweight}", 
            f"--minsize={minsize}", 
            f"--maxsize={maxsize}", 
            f"--seedrand={seedrand}"
            ]
  return command

def generateTest(num_items, minweight, maxweight, minsize, maxsize, seedrand = 0):
  try:
    result = subprocess.run(testGenerateCommand(num_items, minweight, maxweight, minsize, maxsize, seedrand), capture_output=True, text=True, check=True)
    return result.stdout
  except:
    print("Error happened when trying to generate a test instance. Quitting.")
    exit(1)

def testAlgo(program, test_instance):
  try:
    start_time = time.time()
    result = subprocess.run(program, input=test_instance, encoding='utf-8', capture_output=True, check=True)
    end_time = time.time()
    return (result.stdout.strip(), end_time - start_time)
  except Exception as e:
    print(f"Error happened when trying to process a test instance. {program} | {e} | Quitting.")
    exit(1)

def perform():  
  for idx, num_items in enumerate(num_items_list):
    labels = []
    dynamicTimes = []
    dynamicScores = []
    bruteTimes = []
    bruteParTimes = []
    minizincTimes = []
    geneticTimes = []
    geneticScores = []
    geneticParTimes = []
    geneticParScores = []
    greedyTimes = []
    greedyScores = []
    greedyParTimes = []
    greedyParScores = []
    i = 1
    j = 2
    while j < max_range:
      #print(f"Testing {i}, {j}")
      test = generateTest(num_items, i, j, i, j, 0)
      labels.append(f'{i}:{j}')

      bruteScore, bruteTime = testAlgo('./build/bruteKnapsack', test)
      bruteTimes.append(bruteTime)

      dynamicScore, dynamicTime = testAlgo('./build/dynamicKnapsack', test)
      dynamicTimes.append(dynamicTime)
      dynamicScores.append(dynamicScore)

      minizincScore, minizincTime = testAlgo(['python3', 'build/MinizincDriver.py'], test)
      minizincTimes.append(minizincTime)

      greedyScore, greedyTime = testAlgo('./build/greedyKnapsack', test)
      greedyScores.append(greedyScore)
      greedyTimes.append(greedyTime)

      geneticScore, geneticTime = testAlgo('./build/geneticKnapsack', test)
      geneticScores.append(geneticScore)
      geneticTimes.append(geneticTime)
      #print(f"Brute: {testAlgo('./build/bruteKnapsack', test)}")
      #print(f"Dynamic: {testAlgo('./build/dynamicKnapsack', test)}")
      j += 1
    while i < max_range:
      #print(f"Testing {i}, {j}")
      test = generateTest(num_items, i, j, i, j, 0)
      labels.append(f'{i}:{j}')

      bruteScore, bruteTime = testAlgo('./build/bruteKnapsack', test)
      bruteTimes.append(bruteTime)

      bruteParScore, bruteParTime = testAlgo('./build/bruteKnapsackPar', test)
      bruteParTimes.append(bruteParTime)

      dynamicScore, dynamicTime = testAlgo('./build/dynamicKnapsack', test)
      dynamicTimes.append(dynamicTime)
      dynamicScores.append(dynamicScore)

      # minizincScore, minizincTime = testAlgo(['python3', 'build/MinizincDriver.py'], test)
      # minizincTimes.append(minizincTime)

      greedyScore, greedyTime = testAlgo('./build/greedyKnapsack', test)
      greedyScores.append(greedyScore)
      greedyTimes.append(greedyTime)

      greedyParScore, greedyParTime = testAlgo('./build/greedyKnapsackPar', test)
      greedyParScores.append(greedyParScore)
      greedyParTimes.append(greedyParTime)

      geneticScore, geneticTime = testAlgo('./build/geneticKnapsack', test)
      geneticScores.append(geneticScore)
      geneticTimes.append(geneticTime)
      #print(f"Brute: {testAlgo('./build/bruteKnapsack', test)}")
      #print(f"Dynamic: {testAlgo('./build/dynamicKnapsack', test)}")
      i += 1
    x = [i for i in range(1, len(labels) + 1)]
    plt.subplot(len(num_items_list), 2, 2 * idx + 1)
    plt.title(f"Algorytmy dokładne dla $n = {num_items}$")
    plt.plot(x, bruteTimes, label = "Bruteforce")
    plt.plot(x, dynamicTimes, label = "Dynamic")
    plt.plot(x, minizincTimes, label = "Minizinc")
    plt.legend()
    plt.xticks(x[::5], labels[::5], rotation=90)
    plt.subplot(len(num_items_list), 2, 2 * idx + 2)
    plt.title(f"Algorytmy aproksymacyjne dla $n = {num_items}$")
    #plt.plot(x, dynamicScores, label = "Optimal solution")
    geneticRatio = []
    greedyRatio = []
    for i in range(len(greedyScores)):
      geneticRatio.append(int(geneticScores[i]) / int(dynamicScores[i]))
      greedyRatio.append(int(greedyScores[i]) / int(dynamicScores[i]))
    plt.plot(x, geneticRatio, label = "Genetic")
    plt.plot(x, greedyRatio, label = "Greedy")
    plt.legend()
    plt.xticks(x[::5], labels[::5], rotation=90)
  plt.show()
perform()