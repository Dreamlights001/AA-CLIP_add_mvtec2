import os
import subprocess
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Get all datasets from constants.py
DATASETS = [
    "Brain",
    "Liver",
    "Retina",
    "Colon_clinicDB",
    "Colon_colonDB",
    "Colon_cvc300",
    "Colon_Kvasir",
    "BTAD",
    "MPDD",
    "MVTec",
    "VisA",
    "MVTec2"
]

def run_test(dataset, save_path, results_dir):
    """Run test for a single dataset"""
    print(f"Testing {dataset}...")
    
    # Run test command
    cmd = [
        "python", "test.py",
        "--dataset", dataset,
        "--save_path", save_path
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    # Create dataset-specific result directory
    dataset_result_dir = os.path.join(results_dir, dataset)
    os.makedirs(dataset_result_dir, exist_ok=True)
    
    # Save test log
    log_path = os.path.join(dataset_result_dir, "test.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Test command: {' '.join(cmd)}\n")
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        f.write(f"Duration: {end_time - start_time:.2f} seconds\n\n")
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)
    
    print(f"Test completed for {dataset} in {end_time - start_time:.2f} seconds")
    print(f"Results saved to: {log_path}")
    
    return dataset, result.returncode

def main():
    parser = argparse.ArgumentParser(description="Test all datasets automatically")
    parser.add_argument("--save_path", type=str, default="ckpt/baseline", help="Model save path")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save test results")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel tests")
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Results will be saved to: {args.results_dir}")
    
    # Test all datasets
    results = []
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all test jobs
        future_to_dataset = {
            executor.submit(run_test, dataset, args.save_path, args.results_dir): dataset
            for dataset in DATASETS
        }
        
        # Collect results
        for future in as_completed(future_to_dataset):
            dataset = future_to_dataset[future]
            try:
                dataset_name, returncode = future.result()
                results.append((dataset_name, returncode))
            except Exception as exc:
                print(f"{dataset} generated an exception: {exc}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    success_count = 0
    failure_count = 0
    
    for dataset, returncode in results:
        status = "SUCCESS" if returncode == 0 else "FAILED"
        print(f"{dataset}: {status}")
        if returncode == 0:
            success_count += 1
        else:
            failure_count += 1
    
    print("="*60)
    print(f"Total datasets: {len(DATASETS)}")
    print(f"Successful tests: {success_count}")
    print(f"Failed tests: {failure_count}")
    print("="*60)
    
    # Save summary
    summary_path = os.path.join(args.results_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("TEST SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Total datasets: {len(DATASETS)}\n")
        f.write(f"Successful tests: {success_count}")
        f.write(f"Failed tests: {failure_count}\n")
        f.write("="*60 + "\n")
        f.write("Detailed results:\n")
        for dataset, returncode in results:
            status = "SUCCESS" if returncode == 0 else "FAILED"
            f.write(f"{dataset}: {status}\n")
    
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
