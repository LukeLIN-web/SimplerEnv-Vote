import os
import subprocess
from pathlib import Path
from multiprocessing import Process
from datetime import datetime

def run_evaluation_on_dir(log_dir):
    print(f"current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🚀 Running metric calculation for: {log_dir}")

    with open(f"{log_dir}/total.metrics", "a") as f:
        subprocess.run(
            ["python", "tools/calc_metrics_evaluation_videos.py", "--log-dir-root", str(log_dir)],
            stdout=f
        )

    print(f"✅ DONE: {log_dir}")
    print(f"current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def launch_eval_in_process(log_dir):
    p = Process(target=run_evaluation_on_dir, args=(log_dir,))
    p.start()
    return p


if __name__ == "__main__":
    results_root = Path("results")
    all_subdirs = [d for d in results_root.iterdir() if d.is_dir()]
    
    print(f"Found {len(all_subdirs)} result directories.")

    processes = []
    for log_dir in all_subdirs:
        print(f"Starting evaluation for {log_dir}")
        p = launch_eval_in_process(log_dir)
        processes.append(p)