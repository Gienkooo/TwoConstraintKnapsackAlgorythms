import csv 
import os
import json
import subprocess
import re
import multiprocessing
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys

build_config = "Release" 
output_dir = "results"   
num_runs_per_case = 3    
timeout_seconds = 600    
max_threads = multiprocessing.cpu_count() 
thread_list_scalability = [1] + list(range(2, max_threads + 1, 2)) 

max_range = 50 
min_range_start = 20 

n_list_stage1 = [15, 18, 20, 22, 24, 26] 
algos_stage1 = [
    'bruteKnapsack',      
    'bruteKnapsackPar',   
    'bruteKnapsackCuda',  
    'dynamicKnapsack',    
    'dynamicKnapsackCuda',
    'greedyKnapsack',     
    'greedyKnapsackPar',  
    'greedyKnapsackCuda', 
    'geneticKnapsack',    
    'geneticKnapsackPar', 
    'geneticKnapsackCuda' 
]

n_list_stage2 = [50, 100, 200, 500, 1000, 2000] 
algos_stage2 = [
    'greedyKnapsack',     
    'greedyKnapsackPar',  
    'greedyKnapsackCuda', 
    'geneticKnapsack',    
    'geneticKnapsackPar', 
    'geneticKnapsackCuda' 
]

n_for_scalability = 28 
algos_scalability = [ 
    'bruteKnapsackPar',
    'greedyKnapsackPar',
    'geneticKnapsackPar',
]

seq_map_scalability = {
    'bruteKnapsackPar': 'bruteKnapsack',
    'greedyKnapsackPar': 'greedyKnapsack',
    'geneticKnapsackPar': 'geneticKnapsack'
}

all_raw_data = []

build_path = f"./build/{build_config}" 
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
csv_filename = os.path.join(output_dir, f"benchmark_data_{timestamp}.csv") 
csv_header = ['stage', 'n', 'algorithm', 'num_threads', 'avg_time_sec', 'run_times_sec', 'value', 'weight1', 'weight2', 'solution_valid']



def testGenerateCommand(num_items, min_val, max_val, seedrand=0):
    """Creates the command list for the test generation executable."""
    command = [os.path.join(build_path, "testgen"),
               f"--numitems={num_items}",
               f"--minweight={min_val}",
               f"--maxweight={max_val}", 
               f"--minsize={min_val}",
               f"--maxsize={max_val}",   
               f"--maxvalue={max_val}",  
               f"--seedrand={seedrand}"
              ]
    return command

def generateTest(num_items, min_val, max_val, seedrand=0):
    """Runs the test generator and returns the test instance JSON string."""
    process_run_result = None  
    try:
        command = testGenerateCommand(num_items, min_val, max_val, seedrand)
        process_run_result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        if not process_run_result.stdout.strip():
            print(f"Error: testgen produced empty output for command: {' '.join(command)}")
            if process_run_result.stderr:
                print(f"Stderr: {process_run_result.stderr.strip()}")
            return None
        
        json.loads(process_run_result.stdout)
        return process_run_result.stdout
    except json.JSONDecodeError as e:
        print(f"Error: testgen produced invalid JSON: {e}")
        if process_run_result and hasattr(process_run_result, 'stdout') and process_run_result.stdout:
            print(f"Output from testgen (first 200 chars): {process_run_result.stdout[:200]}...")
        else:
            print("Output from testgen is unavailable or was empty.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error generating test instance: {e}")
        if hasattr(e, 'stdout') and e.stdout:
            print(f"Stdout: {e.stdout.strip()}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Stderr: {e.stderr.strip()}")
        return None
    except FileNotFoundError:
        
        print(f"Error: testgen executable not found at {os.path.join(build_path, 'testgen')}")
        sys.exit(1) 
    except Exception as e:
        print(f"An unexpected error occurred during test generation: {e}")
        if process_run_result and hasattr(process_run_result, 'stdout') and process_run_result.stdout:
            print(f"Partial output from testgen (if any, first 200 chars): {process_run_result.stdout[:200]}...")
        return None

def parse_cpp_time(stderr_output):
    """Parses the timing string 'Time: X ms' from C++ stderr."""
    
    match = re.search(r"Time:\s*([\d.]+)\s*ms", stderr_output)
    if match:
        try:
            return float(match.group(1)) / 1000.0
        except ValueError:
            return None
    return None

def run_single_test(program_name, test_instance_str, num_threads):
    """Runs a single instance of a knapsack algorithm."""
    program_path = os.path.join(build_path, program_name)
    run_time_sec = None
    max_value_achieved = -1
    total_weight1 = -1
    total_weight2 = -1
    is_valid_solution = False 

    try:
        test_data_dict = json.loads(test_instance_str)

        if not program_name.endswith("Cuda"):
            test_data_dict["num_threads"] = num_threads
        
        modified_test_instance_str = json.dumps(test_data_dict)

        result = subprocess.run([program_path], input=modified_test_instance_str, encoding='utf-8',
                                capture_output=True, check=True, timeout=timeout_seconds)

        run_time_sec = parse_cpp_time(result.stderr)

        
        if run_time_sec is None and program_name.endswith("Cuda"):
            match_cuda = re.search(r"CUDA Time:\\s*([\\d.]+)\\s*ms", result.stderr)
            if match_cuda:
                try:
                    run_time_sec = float(match_cuda.group(1)) / 1000.0
                except ValueError:
                    pass 

        if run_time_sec is None:
            print(f"Warning: Could not parse time for {program_name} (T={num_threads}). Stderr: {result.stderr.strip()}")

        try:
            solution_details = json.loads(result.stdout)
            max_value_achieved = solution_details.get("value", -1)
            total_weight1 = solution_details.get("weight1", -1) 
            total_weight2 = solution_details.get("weight2", -1)
            
            if "weights" in solution_details and isinstance(solution_details["weights"], list) and len(solution_details["weights"]) == 2:
                total_weight1 = solution_details["weights"][0]
                total_weight2 = solution_details["weights"][1]

            is_valid_solution = solution_details.get("valid", False) 
        except json.JSONDecodeError:
            
            
            value_match = re.search(r"Value:\\s*(\\d+)", result.stdout) 
            if value_match:
                max_value_achieved = int(value_match.group(1))
            

    except json.JSONDecodeError:
        print(f"Error: Input JSON invalid for {program_name}. Input: {test_instance_str[:100]}...")
        
    except subprocess.TimeoutExpired:
        print(f"Error: {program_name} (T={num_threads}) timed out after {timeout_seconds}s.")
        run_time_sec = timeout_seconds 
    except subprocess.CalledProcessError as e:
        print(f"Error running {program_name} (T={num_threads}): {e}\\nStderr: {e.stderr.strip()}")
        
    except FileNotFoundError:
        print(f"Error: Algorithm executable not found at {program_path}")
        sys.exit(1) 
    except Exception as e:
        print(f"An unexpected error occurred running {program_name} (T={num_threads}): {e}")
        

    return run_time_sec, max_value_achieved, total_weight1, total_weight2, is_valid_solution


def run_benchmark_stage(stage_name, n_list, algorithms_to_run, current_all_raw_data):
    """Runs benchmarks for Stages 1 and 2 (varying n)."""
    print(f"\\n===== Running Benchmark Stage: {stage_name} =====")

    stage_results_for_plotting = {algo: {n_val: {'avg_time': 0, 'times': [], 'avg_value': 0, 'avg_w1':0, 'avg_w2':0, 'valid_count':0} for n_val in n_list} for algo in algorithms_to_run}

    for n_val in n_list:
        print(f"\\n--- Testing n = {n_val} ---")
        min_val_gen = min_range_start
        max_val_gen = max_range

        for algo_name in algorithms_to_run:
            print(f"  Algorithm: {algo_name}")
            times_for_algo_n = []
            values_for_algo_n = []
            w1s_for_algo_n = []
            w2s_for_algo_n = []
            valid_solutions_count = 0
            
            threads_for_run = 1
            if algo_name.endswith("Par"): 
                threads_for_run = max_threads 
            

            for run_idx in range(num_runs_per_case):
                seed = run_idx 
                print(f"    Run {run_idx + 1}/{num_runs_per_case} (seed={seed})...")
                test_instance = generateTest(n_val, min_val_gen, max_val_gen, seed)
                if not test_instance:
                    print(f"      Skipping run due to test generation error.")
                    continue

                time_sec, val, w1, w2, is_valid = run_single_test(algo_name, test_instance, threads_for_run)

                if time_sec is not None:
                    times_for_algo_n.append(time_sec)
                if val != -1 : 
                    values_for_algo_n.append(val)
                if w1 != -1: w1s_for_algo_n.append(w1)
                if w2 != -1: w2s_for_algo_n.append(w2)
                if is_valid: valid_solutions_count +=1
                
                
                current_all_raw_data.append({
                    'stage': stage_name, 'n': n_val, 'algorithm': algo_name,
                    'num_threads': threads_for_run, 
                    'avg_time_sec': time_sec if time_sec is not None else 'Error/Timeout', 
                    'run_times_sec': f"[{time_sec}]" if time_sec is not None else 'Error/Timeout', 
                    'value': val, 'weight1': w1, 'weight2': w2,
                    'solution_valid': is_valid
                })

            if times_for_algo_n:
                avg_time = sum(times_for_algo_n) / len(times_for_algo_n)
                stage_results_for_plotting[algo_name][n_val]['avg_time'] = avg_time
                stage_results_for_plotting[algo_name][n_val]['times'] = times_for_algo_n
            if values_for_algo_n:
                 stage_results_for_plotting[algo_name][n_val]['avg_value'] = sum(values_for_algo_n) / len(values_for_algo_n)
            if w1s_for_algo_n: stage_results_for_plotting[algo_name][n_val]['avg_w1'] = sum(w1s_for_algo_n) / len(w1s_for_algo_n)
            if w2s_for_algo_n: stage_results_for_plotting[algo_name][n_val]['avg_w2'] = sum(w2s_for_algo_n) / len(w2s_for_algo_n)
            stage_results_for_plotting[algo_name][n_val]['valid_count'] = valid_solutions_count
            
            print(f"    Avg time for {algo_name} at n={n_val}: {stage_results_for_plotting[algo_name][n_val]['avg_time']:.4f}s over {len(times_for_algo_n)} valid runs")

    return stage_results_for_plotting


def run_scalability_benchmark(stage_name, n_fixed, thread_list, algorithms_to_test, seq_map, current_all_raw_data):
    """Runs scalability tests for Stage 3 (fixed n, varying threads)."""
    print(f"\\n===== Running Benchmark Stage: {stage_name} (n={n_fixed}) =====")

    scalability_results_for_plotting = {algo: {thr: {'avg_time': 0, 'times': [], 'avg_value': 0, 'avg_w1':0, 'avg_w2':0, 'valid_count':0} for thr in thread_list} for algo in algorithms_to_test}
    
    seq_times = {}
    for par_algo, seq_algo in seq_map.items():
        if seq_algo not in algorithms_to_test and par_algo in algorithms_to_test : 
            print(f"  Getting sequential baseline for {par_algo} using {seq_algo} at n={n_fixed}...")
            times_for_seq_n = []
            
            test_instance = generateTest(n_fixed, min_range_start, max_range, seedrand=0) 
            if not test_instance:
                print(f"    Skipping sequential baseline for {seq_algo} due to test generation error.")
                continue
            for run_idx in range(num_runs_per_case):
                time_sec, _, _, _, _ = run_single_test(seq_algo, test_instance, 1) 
                if time_sec is not None:
                    times_for_seq_n.append(time_sec)
            if times_for_seq_n:
                seq_times[par_algo] = sum(times_for_seq_n) / len(times_for_seq_n)
                print(f"    Avg sequential time for {seq_algo} (baseline for {par_algo}): {seq_times[par_algo]:.4f}s")


    for algo_name in algorithms_to_test:
        print(f"\\n--- Algorithm: {algo_name} (n={n_fixed}) ---")
        for num_threads in thread_list:
            if not algo_name.endswith("Par") and num_threads > 1: 
                continue

            print(f"  Testing with {num_threads} threads...")
            times_for_algo_threads = []
            values_for_algo_threads = []
            w1s_for_algo_threads = []
            w2s_for_algo_threads = []
            valid_solutions_count = 0

            for run_idx in range(num_runs_per_case):
                seed = run_idx 
                print(f"    Run {run_idx + 1}/{num_runs_per_case} (seed={seed})...")
                test_instance = generateTest(n_fixed, min_range_start, max_range, seed)
                if not test_instance:
                    print(f"      Skipping run due to test generation error.")
                    continue
                
                time_sec, val, w1, w2, is_valid = run_single_test(algo_name, test_instance, num_threads)
                if time_sec is not None:
                    times_for_algo_threads.append(time_sec)
                if val != -1: values_for_algo_threads.append(val)
                if w1 != -1: w1s_for_algo_threads.append(w1)
                if w2 != -1: w2s_for_algo_threads.append(w2)
                if is_valid: valid_solutions_count +=1

                current_all_raw_data.append({
                    'stage': stage_name, 'n': n_fixed, 'algorithm': algo_name,
                    'num_threads': num_threads,
                    'avg_time_sec': time_sec if time_sec is not None else 'Error/Timeout',
                    'run_times_sec': f"[{time_sec}]" if time_sec is not None else 'Error/Timeout',
                    'value': val, 'weight1': w1, 'weight2': w2,
                    'solution_valid': is_valid
                })
            
            if times_for_algo_threads:
                avg_time = sum(times_for_algo_threads) / len(times_for_algo_threads)
                scalability_results_for_plotting[algo_name][num_threads]['avg_time'] = avg_time
                scalability_results_for_plotting[algo_name][num_threads]['times'] = times_for_algo_threads
            if values_for_algo_threads:
                scalability_results_for_plotting[algo_name][num_threads]['avg_value'] = sum(values_for_algo_threads) / len(values_for_algo_threads)
            
            scalability_results_for_plotting[algo_name][num_threads]['valid_count'] = valid_solutions_count

            print(f"    Avg time for {algo_name} with {num_threads} threads: {scalability_results_for_plotting[algo_name][num_threads]['avg_time']:.4f}s over {len(times_for_algo_threads)} valid runs")
    
    
    if hasattr(scalability_results_for_plotting, 'setdefault'): 
         scalability_results_for_plotting.setdefault('_seq_times', seq_times)


    return scalability_results_for_plotting

def get_plot_style(algo_name):
    """Returns a consistent style for a given algorithm name for plotting."""
    
    style = {'marker': 'o', 'linestyle': '-'}
    if algo_name.endswith("Cuda"):
        style['marker'] = '^'
        style['linestyle'] = '--'
        if "brute" in algo_name: style['color'] = 'red'
        elif "dynamic" in algo_name: style['color'] = 'green'
        elif "greedy" in algo_name: style['color'] = 'blue'
        elif "genetic" in algo_name: style['color'] = 'purple'
        else: style['color'] = 'black' 
    elif algo_name.endswith("Par"):
        style['marker'] = 's'
        style['linestyle'] = ':'
        if "brute" in algo_name: style['color'] = 'darkred'
        elif "dynamic" in algo_name: style['color'] = 'darkgreen' 
        elif "greedy" in algo_name: style['color'] = 'darkblue'
        elif "genetic" in algo_name: style['color'] = 'indigo'
        else: style['color'] = 'gray' 
    else: 
        style['marker'] = 'o'
        style['linestyle'] = '-'
        if "brute" in algo_name: style['color'] = 'salmon'
        elif "dynamic" in algo_name: style['color'] = 'lightgreen'
        elif "greedy" in algo_name: style['color'] = 'lightblue'
        elif "genetic" in algo_name: style['color'] = 'violet'
        else: style['color'] = 'dimgray' 
    return style

def plot_stage1_results(data, n_values, algorithms, output_directory, time_stamp_str):
    plt.figure(figsize=(12, 8))
    for algo in algorithms:
        if algo not in data: continue
        avg_times = [data[algo].get(n, {}).get('avg_time', float('nan')) for n in n_values]
        style = get_plot_style(algo)
        plt.plot(n_values, avg_times, label=algo, marker=style['marker'], linestyle=style['linestyle'], color=style['color'])
    plt.xlabel("Number of Items (n)")
    plt.ylabel("Average Runtime (seconds)")
    plt.title("Stage 1: Runtime vs. N (Small N)")
    plt.legend()
    plt.grid(True)
    plt.yscale('log') 
    plot_filename = os.path.join(output_directory, f"plot_stage1_runtime_{time_stamp_str}.png")
    plt.savefig(plot_filename)
    print(f"Stage 1 plot saved to {plot_filename}")
    plt.close()

def plot_stage2_results(data, n_values, algorithms, output_directory, time_stamp_str):
    plt.figure(figsize=(12, 8))
    for algo in algorithms:
        if algo not in data: continue
        avg_times = [data[algo].get(n, {}).get('avg_time', float('nan')) for n in n_values]
        style = get_plot_style(algo)
        plt.plot(n_values, avg_times, label=algo, marker=style['marker'], linestyle=style['linestyle'], color=style['color'])
    plt.xlabel("Number of Items (n)")
    plt.ylabel("Average Runtime (seconds)")
    plt.title("Stage 2: Runtime vs. N (Large N)")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plot_filename = os.path.join(output_directory, f"plot_stage2_runtime_{time_stamp_str}.png")
    plt.savefig(plot_filename)
    print(f"Stage 2 plot saved to {plot_filename}")
    plt.close()

def plot_scalability_results(data, n_value_fixed, thread_values, algorithms, seq_data_map, output_directory, time_stamp_str):
    plt.figure(figsize=(12, 8))
    
    ax1 = plt.gca()
    for algo in algorithms: 
        if algo not in data or not algo.endswith("Par"): continue 
        avg_times = [data[algo].get(threads, {}).get('avg_time', float('nan')) for threads in thread_values]
        style = get_plot_style(algo)
        ax1.plot(thread_values, avg_times, label=f"{algo} (abs time)", marker=style['marker'], linestyle=style['linestyle'], color=style['color'])
    
    ax1.set_xlabel("Number of Threads")
    ax1.set_ylabel("Average Runtime (seconds)")
    ax1.set_title(f"Scalability Analysis (n={n_value_fixed})")
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax1.set_yscale('log')

    
    
    seq_times_lookup = data.get('_seq_times', {})
    
    if seq_times_lookup:
        ax2 = ax1.twinx() 
        for algo in algorithms:
            if algo not in data or not algo.endswith("Par"): continue
            if algo not in seq_times_lookup or seq_times_lookup[algo] == 0: continue

            seq_time = seq_times_lookup[algo]
            speedup_values = [seq_time / data[algo].get(threads, {}).get('avg_time', float('inf')) if data[algo].get(threads, {}).get('avg_time') else 0 for threads in thread_values]
            style = get_plot_style(algo) 
            
            ax2.plot(thread_values, speedup_values, label=f"{algo} (speedup)", marker=style['marker'], linestyle='--', color=style['color'], alpha=0.7)

        ax2.set_ylabel("Speedup (Sequential Time / Parallel Time)")
        ax2.legend(loc='center right')
        
        ax2.plot(thread_values, thread_values, linestyle=':', color='gray', label='Ideal Speedup')


    plot_filename = os.path.join(output_directory, f"plot_scalability_n{n_value_fixed}_{time_stamp_str}.png")
    plt.savefig(plot_filename)
    print(f"Scalability plot saved to {plot_filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run knapsack algorithm benchmarks.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stages', nargs='*', type=int, choices=[1, 2, 3], default=None,
                        help='List of stages to run (1, 2, 3). Runs all if not specified or if list is empty.')
    parser.add_argument('--algorithms', nargs='*', type=str, default=None,
                        help='List of specific algorithms to run (e.g., bruteKnapsack greedyKnapsackCuda). Runs all relevant algorithms for the selected stages if not specified.')
    parser.add_argument('--n_values_stage1', nargs='*', type=int, default=None, help='Override n values for Stage 1.')
    parser.add_argument('--n_values_stage2', nargs='*', type=int, default=None, help='Override n values for Stage 2.')
    parser.add_argument('--n_value_scalability', type=int, default=None, help='Override n value for Stage 3 (Scalability).')

    args = parser.parse_args()
    current_all_raw_data = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    stages_to_run = args.stages
    if stages_to_run is None or not stages_to_run: 
        stages_to_run = [1, 2, 3] 

    selected_algorithms_filter = args.algorithms

    current_n_list_stage1 = args.n_values_stage1 if args.n_values_stage1 else n_list_stage1
    current_n_list_stage2 = args.n_values_stage2 if args.n_values_stage2 else n_list_stage2
    current_n_for_scalability = args.n_value_scalability if args.n_value_scalability is not None else n_for_scalability

    if 1 in stages_to_run:
        algos_for_stage1 = algos_stage1
        if selected_algorithms_filter:
            algos_for_stage1 = [a for a in algos_stage1 if a in selected_algorithms_filter]
        
        if not algos_for_stage1:
            print("No algorithms selected or available for Stage 1 based on current filters.")
        else:
            results_stage1 = run_benchmark_stage("Stage 1 (Small N)", current_n_list_stage1, algos_for_stage1, current_all_raw_data)
            if results_stage1: 
                plot_stage1_results(results_stage1, current_n_list_stage1, algos_for_stage1, output_dir, timestamp)

    
    if 2 in stages_to_run:
        algos_for_stage2 = algos_stage2
        if selected_algorithms_filter:
            algos_for_stage2 = [a for a in algos_stage2 if a in selected_algorithms_filter]

        if not algos_for_stage2:
            print("No algorithms selected or available for Stage 2 based on current filters.")
        else:
            results_stage2 = run_benchmark_stage("Stage 2 (Large N)", current_n_list_stage2, algos_for_stage2, current_all_raw_data)
            if results_stage2:
                plot_stage2_results(results_stage2, current_n_list_stage2, algos_for_stage2, output_dir, timestamp)
                
    
    if 3 in stages_to_run:
        algos_for_scalability_stage = algos_scalability
        current_seq_map = seq_map_scalability
        if selected_algorithms_filter:
            algos_for_scalability_stage = [a for a in algos_scalability if a in selected_algorithms_filter]
            current_seq_map = {k: v for k, v in seq_map_scalability.items() if k in algos_for_scalability_stage}

        if not algos_for_scalability_stage:
            print("No algorithms selected or available for Stage 3 based on current filters.")
        else:
            results_scalability = run_scalability_benchmark("Stage 3 (Scalability)", current_n_for_scalability, thread_list_scalability, algos_for_scalability_stage, current_seq_map, current_all_raw_data)
            if results_scalability:
                 plot_scalability_results(results_scalability, current_n_for_scalability, thread_list_scalability, algos_for_scalability_stage, current_seq_map, output_dir, timestamp)

    
    if current_all_raw_data:
        
        
        final_csv_filename = os.path.join(output_dir, f"benchmark_data_{timestamp}.csv") 
        try:
            with open(final_csv_filename, 'w', newline='') as f:
                if current_all_raw_data:
                    actual_header = list(current_all_raw_data[0].keys())
                    for key_to_check in ['value', 'weight1', 'weight2', 'solution_valid']:
                        if key_to_check not in actual_header and any(item.get(key_to_check) is not None for item in current_all_raw_data):
                             print(f"Warning: CSV header might be missing key: {key_to_check}")
                    writer = csv.DictWriter(f, fieldnames=csv_header)
                else: 
                    writer = csv.DictWriter(f, fieldnames=csv_header)

                writer.writeheader()
                writer.writerows(current_all_raw_data)
            print(f"\\nBenchmark data saved to {final_csv_filename}")
        except Exception as e:
            print(f"Error writing CSV file: {e}")
    else:
        print("\\nNo benchmark data collected.")

if __name__ == "__main__":
    main()