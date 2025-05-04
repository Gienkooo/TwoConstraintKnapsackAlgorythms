import subprocess
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import os # Import os module for path operations
import csv # Import csv module
from datetime import datetime # Import datetime module
import re # Import regex module for parsing stderr
import multiprocessing # For getting CPU count

# --- Configuration ---
# General
build_config = "Release" # Or "Debug"
output_dir = "results"   # Directory to save plots and data
num_runs_per_case = 3    # Number of runs to average
timeout_seconds = 600    # Timeout per algorithm run
max_threads = multiprocessing.cpu_count() # Use detected core count
thread_list_scalability = [1] + list(range(2, max_threads + 1, 2)) # e.g., [1, 2, 4, ..., max_threads]

# Test Generation
max_range = 50 # Max item weight/size/value
min_range_start = 20 # Min item weight/size/value

# Stage 1: Small 'n' Comparison (Brute, Dynamic, Greedy, Genetic)
n_list_stage1 = [15, 18, 20, 22, 24, 26] # n values up to ~24
algos_stage1 = [
    'bruteKnapsack',      # Sequential Brute Force
    'bruteKnapsackPar',   # Parallel Brute Force
    'dynamicKnapsack',    # Dynamic Programming (Optimal)
    'greedyKnapsack',     # Sequential Greedy
    'greedyKnapsackPar',  # Parallel Greedy
    'geneticKnapsack',    # Sequential Genetic
    'geneticKnapsackPar'  # Parallel Genetic
]

# Stage 2: Large 'n' Comparison (Greedy, Genetic)
n_list_stage2 = [50, 100, 200, 500, 1000, 2000] # n values beyond Brute Force
algos_stage2 = [
    'greedyKnapsack',     # Sequential Greedy
    'greedyKnapsackPar',  # Parallel Greedy
    'geneticKnapsack',    # Sequential Genetic
    'geneticKnapsackPar'  # Parallel Genetic
]

# Stage 3: Scalability Analysis
# Note: n=24 is small for Greedy/Genetic scaling, consider increasing if needed
n_for_scalability = 28
algos_scalability = [ # Parallel algorithms to test
    'bruteKnapsackPar',
    'greedyKnapsackPar',
    'geneticKnapsackPar'
]
# Map parallel algos to their sequential counterparts for overhead comparison
seq_map_scalability = {
    'bruteKnapsackPar': 'bruteKnapsack',
    'greedyKnapsackPar': 'greedyKnapsack',
    'geneticKnapsackPar': 'geneticKnapsack'
}

# --- Setup ---
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
build_path = f"./build/{build_config}"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = os.path.join(output_dir, f"benchmark_data_{timestamp}.csv")
csv_header = ['stage', 'n', 'algorithm', 'num_threads', 'avg_time_sec', 'run_times_sec']
all_raw_data = [] # Collect data across all stages for single CSV write

# --- Helper Functions (testGenerateCommand, generateTest, parse_cpp_time, testAlgo) ---
# Keep these functions as they were in the previous version (including timeout in testAlgo)
def testGenerateCommand(num_items, min_val, max_val, seedrand=0):
    """Creates the command list for the test generation executable."""
    command = [os.path.join(build_path, "testgen"),
               f"--numitems={num_items}",
               f"--minweight={min_val}",
               f"--maxweight={max_val}", # Capacity W tied to max item weight
               f"--minsize={min_val}",
               f"--maxsize={max_val}",   # Capacity S tied to max item size
               f"--maxvalue={max_val}",  # Max item value also tied
               f"--seedrand={seedrand}"
              ]
    return command

def generateTest(num_items, min_val, max_val, seedrand=0):
    """Runs the test generator and returns the test instance JSON string."""
    try:
        command = testGenerateCommand(num_items, min_val, max_val, seedrand)
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        if not result.stdout.strip():
            print(f"Error: testgen produced empty output for command: {' '.join(command)}")
            print(f"Stderr: {result.stderr}")
            return None
        # Validate JSON structure before returning
        json.loads(result.stdout)
        return result.stdout
    except json.JSONDecodeError as e:
        print(f"Error: testgen produced invalid JSON: {e}")
        print(f"Output: {result.stdout[:200]}...")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error generating test instance: {e}\nStderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"Error: testgen executable not found at {os.path.join(build_path, 'testgen')}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during test generation: {e}")
        return None

def parse_cpp_time(stderr_output):
    """Parses the timing string 'Time: X ms' from C++ stderr."""
    # Regex captures digits and dots, handling floating point ms values
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
    try:
        test_data = json.loads(test_instance_str)
        test_data["num_threads"] = num_threads
        modified_test_instance = json.dumps(test_data)

        result = subprocess.run([program_path], input=modified_test_instance, encoding='utf-8',
                                capture_output=True, check=True, timeout=timeout_seconds)

        # We primarily care about time now
        algo_time_sec = parse_cpp_time(result.stderr)
        if algo_time_sec is None:
            print(f"Warning: Could not parse time for {program_name} (T={num_threads}). Stderr: {result.stderr.strip()}")
            return None # Indicate failure

        return algo_time_sec # Return time in seconds

    except json.JSONDecodeError:
        print(f"Error: Input JSON invalid for {program_name}. Input: {test_instance_str[:100]}...")
        return None
    except subprocess.TimeoutExpired:
        print(f"Error: {program_name} (T={num_threads}) timed out after {timeout_seconds}s.")
        return timeout_seconds # Indicate timeout with max time
    except subprocess.CalledProcessError as e:
        print(f"Error running {program_name} (T={num_threads}): {e}\nStderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"Error: Algorithm executable not found at {program_path}")
        # This is fatal, maybe exit? For now, return None.
        return None
    except Exception as e:
        print(f"An unexpected error occurred running {program_name} (T={num_threads}): {e}")
        return None

# --- Benchmarking Functions ---
def run_benchmark_stage(stage_name, n_list, algorithms_to_run):
    """Runs benchmarks for Stages 1 and 2 (varying n)."""
    print(f"\n===== Running Benchmark Stage: {stage_name} =====")
    # Results structure: {algo_name: {n: [list_of_times]}}
    results = {algo: {n: [] for n in n_list} for algo in algorithms_to_run}

    for n in n_list:
        print(f"\n--- Testing n = {n} ---")
        min_val = min_range_start
        max_val = max_range # Use full range for generation

        valid_runs_for_n = 0
        for run_index in range(num_runs_per_case):
            seed = run_index # Use run_index for reproducibility
            print(f"  Run {run_index + 1}/{num_runs_per_case}...")
            test_instance = generateTest(n, min_val, max_val, seed)
            if not test_instance:
                print(f"    Skipping run {run_index+1} due to test generation error.")
                continue

            run_successful = True
            temp_run_times = {algo: -1.0 for algo in algorithms_to_run} # Store times for this run

            for algo_name in algorithms_to_run:
                # Determine threads: Max for parallel, 1 for sequential
                is_parallel = algo_name.endswith("Par")
                threads_for_run = max_threads if is_parallel else 1

                algo_time_sec = run_single_test(algo_name, test_instance, threads_for_run)

                if algo_time_sec is None or algo_time_sec >= timeout_seconds:
                    print(f"    Run {run_index+1} for {algo_name} (T={threads_for_run}) failed or timed out. Skipping this entire run.")
                    run_successful = False
                    break # Stop processing this run_index

                temp_run_times[algo_name] = algo_time_sec

            if run_successful:
                valid_runs_for_n += 1
                # Append successful run times to results
                for algo_name, time_sec in temp_run_times.items():
                    results[algo_name][n].append(time_sec)
            # else: times for this run_index are discarded

        if valid_runs_for_n < num_runs_per_case:
            print(f"  Warning: n={n} completed only {valid_runs_for_n}/{num_runs_per_case} valid runs.")
        if valid_runs_for_n == 0:
            print(f"  ERROR: n={n} had no valid runs. Data for this 'n' will be missing.")

    # Process results for CSV and return structure for plotting
    processed_results = {algo: {} for algo in algorithms_to_run}
    for algo_name, n_data in results.items():
        is_parallel = algo_name.endswith("Par")
        threads_used = max_threads if is_parallel else 1
        for n, times in n_data.items():
            if times: # Check if list is not empty
                avg_time = np.mean(times)
                processed_results[algo_name][n] = avg_time
                all_raw_data.append({
                    'stage': stage_name, 'n': n, 'algorithm': algo_name,
                    'num_threads': threads_used, 'avg_time_sec': avg_time,
                    'run_times_sec': json.dumps(times) # Store raw times as JSON string
                })
            else:
                 processed_results[algo_name][n] = np.nan # Indicate missing data

    return processed_results

def run_scalability_benchmark(stage_name, n_fixed, thread_list, algorithms_to_test, seq_map):
    """Runs scalability tests for Stage 3 (fixed n, varying threads)."""
    print(f"\n===== Running Benchmark Stage: {stage_name} (n={n_fixed}) =====")
    # Results structure: {algo_name: {num_threads: [list_of_times]}}
    results = {algo: {t: [] for t in thread_list} for algo in algorithms_to_test}
    # Add entries for sequential baselines
    seq_algorithms = [seq_map[p] for p in algorithms_to_test if p in seq_map and seq_map[p]]
    results.update({seq_algo: {1: []} for seq_algo in seq_algorithms}) # Seq algos run with T=1

    min_val = min_range_start
    max_val = max_range
    print(f"--- Generating fixed test instance for n = {n_fixed} ---")
    fixed_seed = 42
    test_instance = generateTest(n_fixed, min_val, max_val, fixed_seed)
    if not test_instance:
        print(f"FATAL ERROR: Could not generate test instance for scalability test (n={n_fixed}). Aborting stage.")
        return None

    print(f"--- Testing algorithms with threads: {thread_list} ---")

    # Run Sequential Baselines first
    for seq_algo_name in seq_algorithms:
         print(f"  Running Sequential Baseline: {seq_algo_name} (x{num_runs_per_case} runs)...")
         for run_index in range(num_runs_per_case):
             algo_time_sec = run_single_test(seq_algo_name, test_instance, num_threads=1)
             if algo_time_sec is not None:
                 results[seq_algo_name][1].append(algo_time_sec)
             else:
                 print(f"    Run {run_index+1} for {seq_algo_name} failed. Data point missing.")

    # Run Parallel Algorithms
    for algo_name in algorithms_to_test:
        print(f"  Testing Parallel Algorithm: {algo_name}")
        for num_threads in thread_list:
            print(f"    Threads = {num_threads} (x{num_runs_per_case} runs)...")
            for run_index in range(num_runs_per_case):
                algo_time_sec = run_single_test(algo_name, test_instance, num_threads=num_threads)

                if algo_time_sec is None or algo_time_sec >= timeout_seconds:
                    print(f"      Run {run_index+1} for T={num_threads} failed or timed out. Skipping run.")
                    continue # Skip this run, try next

                results[algo_name][num_threads].append(algo_time_sec)

            if not results[algo_name][num_threads]:
                 print(f"    ERROR: T={num_threads} had no valid runs. Data for this thread count will be missing.")

    # Process results for CSV and return structure for plotting
    processed_results = {}
    for algo_name, thread_data in results.items():
        processed_results[algo_name] = {}
        for num_threads, times in thread_data.items():
            if times:
                avg_time = np.mean(times)
                processed_results[algo_name][num_threads] = avg_time
                all_raw_data.append({
                    'stage': stage_name, 'n': n_fixed, 'algorithm': algo_name,
                    'num_threads': num_threads, 'avg_time_sec': avg_time,
                    'run_times_sec': json.dumps(times)
                })
            else:
                processed_results[algo_name][num_threads] = np.nan

    return processed_results

# --- Plotting Functions ---
def plot_stage1_results(results, n_values):
    """Plots Stage 1 results: Runtime vs. n."""
    print(f"\n--- Generating Plot for Stage 1 (n={n_values}) ---")
    plt.figure(figsize=(12, 7)) # Adjusted size slightly
    ax = plt.gca()
    ax.set_title(f"Runtime vs. Number of Items (Stage 1)")

    # Plot data, handling potential missing points (NaN)
    for algo_name, n_data in results.items():
        times = [n_data.get(n, np.nan) for n in n_values]
        valid_indices = ~np.isnan(times)
        valid_n = np.array(n_values)[valid_indices]
        valid_times = np.array(times)[valid_indices]

        if len(valid_times) > 0:
            label = algo_name
            marker = '.'
            linestyle = '-'
            is_parallel = algo_name.endswith("Par")

            # Assign markers/styles based on algorithm type
            if 'brute' in algo_name:
                marker = '.' if not is_parallel else 'x'
            elif 'dynamic' in algo_name:
                marker = '+'
            elif 'greedy' in algo_name:
                marker = 's' if not is_parallel else '^'
            elif 'genetic' in algo_name:
                marker = 'o' if not is_parallel else 'd'

            linestyle = '--' if is_parallel else '-'
            if algo_name == 'dynamicKnapsack': # Special case for dynamic
                 linestyle = ':'

            if is_parallel:
                label += f" (Max T={max_threads})"

            ax.plot(valid_n, valid_times, label=label, marker=marker, linestyle=linestyle)
        else:
            print(f"  Warning: No valid runtime data to plot for {algo_name} in Stage 1.")

    ax.set_ylabel("Average Time (s)")
    ax.set_xlabel("Number of Items (n)")
    ax.legend()
    ax.set_xticks(n_values)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_yscale('log')

    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"plot_stage1_runtime_{timestamp}.png")
    try:
        plt.savefig(plot_filename)
        print(f"Stage 1 plot saved successfully to {plot_filename}")
    except Exception as e:
        print(f"Error saving Stage 1 plot: {e}")
    plt.close()

def plot_stage2_results(results, n_values):
    """Plots Stage 2 results: Runtime vs. n."""
    print(f"\n--- Generating Plot for Stage 2 (n={n_values}) ---")
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_title(f"Runtime vs. Number of Items (Stage 2)")

    for algo_name, n_data in results.items():
        times = [n_data.get(n, np.nan) for n in n_values]
        valid_indices = ~np.isnan(times)
        valid_n = np.array(n_values)[valid_indices]
        valid_times = np.array(times)[valid_indices]

        if len(valid_times) > 0:
            label = algo_name
            marker = '.' if 'Genetic' in algo_name else '+'
            linestyle = '-'
            if algo_name.endswith("Par"):
                label += f" (Max T={max_threads})"
                linestyle = '--'
            ax.plot(valid_n, valid_times, label=label, marker=marker, linestyle=linestyle)
        else:
            print(f"  Warning: No valid runtime data to plot for {algo_name} in Stage 2.")

    ax.set_ylabel("Average Time (s)")
    ax.set_xlabel("Number of Items (n)")
    ax.legend()
    # ax.set_xticks(n_values) # Might be too crowded for large n ranges
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_yscale('log')
    # ax.set_xscale('log') # Optional: Use log-log if expecting polynomial scaling

    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"plot_stage2_runtime_{timestamp}.png")
    try:
        plt.savefig(plot_filename)
        print(f"Stage 2 plot saved successfully to {plot_filename}")
    except Exception as e:
        print(f"Error saving Stage 2 plot: {e}")
    plt.close()

def plot_scalability_results(results, n_fixed, thread_list, seq_map):
    """Generates scalability plots (Runtime vs Threads, Speedup vs Threads)."""
    print(f"\n--- Generating Scalability Plots (n = {n_fixed}) ---")
    parallel_algorithms = [algo for algo in results if algo.endswith("Par")]
    if not parallel_algorithms:
        print("No parallel scalability data to plot.")
        return

    num_plots = 2 # Runtime, Speedup
    plt.figure(figsize=(8 * num_plots, 6))

    # Plot 1: Runtime vs Threads
    ax1 = plt.subplot(1, num_plots, 1)
    ax1.set_title(f"Runtime vs. Number of Threads (n = {n_fixed})")
    ax1.set_xlabel("Number of Threads")
    ax1.set_ylabel("Average Time (s)")
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_xticks(thread_list)

    # Plot 2: Speedup vs Threads
    ax2 = plt.subplot(1, num_plots, 2)
    ax2.set_title(f"Speedup vs. Number of Threads (n = {n_fixed})")
    ax2.set_xlabel("Number of Threads")
    ax2.set_ylabel("Speedup (Time(1 Thread) / Time(T Threads))")
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_xticks(thread_list)

    # Add ideal speedup line
    ax2.plot(thread_list, thread_list, label='Ideal Linear Speedup', color='grey', linestyle=':')

    for algo_name in parallel_algorithms:
        thread_data = results.get(algo_name, {})
        times = [thread_data.get(t, np.nan) for t in thread_list]
        valid_indices = ~np.isnan(times)
        valid_threads = np.array(thread_list)[valid_indices]
        valid_times = np.array(times)[valid_indices]

        if len(valid_times) == 0:
            print(f"  Skipping plots for {algo_name} due to no valid data.")
            continue

        # --- Runtime Plot ---
        ax1.plot(valid_threads, valid_times, label=algo_name, marker='o', linestyle='-')

        # Add sequential baseline to Runtime plot for overhead comparison
        seq_algo_name = seq_map.get(algo_name)
        if seq_algo_name and seq_algo_name in results:
            seq_time = results[seq_algo_name].get(1, np.nan) # Seq time is at T=1
            if not np.isnan(seq_time):
                # Plot as a point or horizontal line
                ax1.plot(1, seq_time, marker='s', markersize=8, linestyle='none',
                         label=f"{seq_algo_name} (Seq Baseline)",
                         color=ax1.lines[-1].get_color()) # Match color of parallel line
                # Or plot as a line across the axis:
                # ax1.axhline(y=seq_time, color=ax1.lines[-1].get_color(), linestyle=':',
                #             label=f"{seq_algo_name} (Seq Baseline)")

        # --- Speedup Plot ---
        baseline_time_par = thread_data.get(1, np.nan) # Speedup relative to T=1 PARALLEL run

        if not np.isnan(baseline_time_par) and baseline_time_par > 1e-9:
            # Calculate speedup only for valid times > 0
            speedup_threads = []
            speedup_values = []
            for i, t_val in enumerate(valid_times):
                if t_val > 1e-9:
                    speedup_threads.append(valid_threads[i])
                    speedup_values.append(baseline_time_par / t_val)

            if speedup_values:
                ax2.plot(speedup_threads, speedup_values, label=f"{algo_name} Speedup", marker='x', linestyle='--')
            else:
                print(f"  Cannot plot speedup for {algo_name}, all valid times were near zero.")
        elif 1 not in valid_threads:
             print(f"  Cannot calculate speedup for {algo_name}, baseline (1 thread parallel) data missing.")
        else:
             print(f"  Cannot calculate speedup for {algo_name}, baseline time (1 thread parallel) is near zero ({baseline_time_par:.2e}s).")

    # Final plot adjustments
    ax1.legend()
    ax1.set_yscale('log') # Runtimes often vary widely
    ax2.legend()

    plt.tight_layout(pad=2.0)
    plot_filename = os.path.join(output_dir, f"plot_scalability_n{n_fixed}_{timestamp}.png")
    try:
        plt.savefig(plot_filename)
        print(f"\nScalability plots saved successfully to {plot_filename}")
    except Exception as e:
        print(f"Error saving scalability plots: {e}")
    plt.close()

def save_results_to_csv(filename):
    """Saves the collected raw data to a CSV file."""
    print(f"\n--- Saving all results to CSV: {filename} ---")
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_header)
            writer.writeheader()
            # Sort data for better readability in CSV (optional)
            sorted_data = sorted(all_raw_data, key=lambda x: (x['stage'], x['algorithm'], x['n'], x['num_threads']))
            writer.writerows(sorted_data)
        print("CSV saving completed successfully.")
    except IOError as e:
        print(f"Error writing CSV file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting benchmarks... Max threads to use for parallel algos: {max_threads}")

    # Run Stage 1
    stage1_results = run_benchmark_stage("Stage1_Small_N", n_list_stage1, algos_stage1)
    if stage1_results:
        plot_stage1_results(stage1_results, n_list_stage1)

    # Run Stage 2
    stage2_results = run_benchmark_stage("Stage2_Large_N", n_list_stage2, algos_stage2)
    if stage2_results:
        plot_stage2_results(stage2_results, n_list_stage2)

    # Run Stage 3 - Scalability
    scalability_results = run_scalability_benchmark("Stage3_Scalability", n_for_scalability,
                                                    thread_list_scalability, algos_scalability,
                                                    seq_map_scalability)
    if scalability_results:
        plot_scalability_results(scalability_results, n_for_scalability,
                                 thread_list_scalability, seq_map_scalability)

    # Save all collected data to CSV
    save_results_to_csv(csv_filename)

    print("\n===== Benchmark Complete =====")
    print(f"Results saved in directory: {output_dir}")
    print(f"Raw data saved to: {csv_filename}")