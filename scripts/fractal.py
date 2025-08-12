import os
import subprocess
import argparse
from pathlib import Path
from multiprocessing import Process
from datetime import datetime

tasks = [
    # "put_in_drawer_visual_matching.sh",
    # "put_in_drawer_variant_agg.sh",
    "pick_coke_can_visual_matching.sh",
    "move_near_visual_matching.sh",
    "drawer_visual_matching.sh",
    "move_near_variant_agg.sh",
    "drawer_variant_agg.sh",
    "pick_coke_can_variant_agg.sh",
]

def run_evaluation(ckpt_path, device, model_name,ensembler):
    
    print(f"current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging_dir = f"results/{Path(ckpt_path).name}_{ensembler}"
    os.makedirs(logging_dir, exist_ok=True)
    
    for task in tasks:
        print(f"🚀 running {task} on GPU {device} for {Path(ckpt_path).name} ...")
        task_log_file = os.path.join(logging_dir, f"{Path(ckpt_path).name}--{task}.log")
        
        with open(task_log_file, "w") as fout, open(task_log_file + ".err", "w") as ferr:
            cmd = [
                "bash",
                f"scripts/{task}",
                ckpt_path,
                model_name,
                '-0.8',
                logging_dir,
                str(device),
                ensembler
            ]
            print(cmd)
            subprocess.run(cmd, stdout=fout, stderr=ferr)
        print(f"current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"🚀 all tasks DONE for {Path(ckpt_path).name} on GPU {device}! Calculating metrics...")
    with open(f"{logging_dir}/total.metrics", "a") as f:
        subprocess.run(
            ["python", "tools/calc_metrics_evaluation_videos.py", "--log-dir-root", logging_dir],
            stdout=f
        )
    print(f"🚀 Calculate metrics... DONE")
    print(f"current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def launch_eval_in_process(ckpt_path, device, model_name,ensembler):
    p = Process(target=run_evaluation, args=(ckpt_path, device, model_name,ensembler))
    p.start()
    return p

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on multiple checkpoints.")
    parser.add_argument("ckpts", nargs="+", help="Paths to checkpoint files")
    parser.add_argument("--devices", type=int, nargs="+", help="GPU device IDs to use (default: 0, 1, 2, ...)")
    parser.add_argument("--model", type=str, default="lisa", help="Model name (default: lisa)") #"openvla"   
    parser.add_argument("--ensembler", type=str, default="vote", help="Ensembler type (default: vote)")
    #在这里默认是adapt.
    args = parser.parse_args()
    
    # 检查路径是否存在或为空文件夹
    for ckpt_path in args.ckpts:
        path = Path(ckpt_path)
        if not path.exists():
            raise ValueError(f"path not exist: {ckpt_path}")
        if path.is_dir() and not any(path.iterdir()):
            raise ValueError(f"path is empty: {ckpt_path}")
    
    # Assign default devices if not specified
    if args.devices is None:
        args.devices = list(range(len(args.ckpts)))
    
    print("we will evaluate, args are ")
    print(args)

    # Ensure we have enough devices for the checkpoints
    if len(args.devices) < len(args.ckpts):
        print(f"Warning: {len(args.ckpts)} checkpoints but only {len(args.devices)} devices specified.")
        print(f"Will only evaluate the first {len(args.devices)} checkpoints.")
        args.ckpts = args.ckpts[:len(args.devices)]
    
    # Launch processes
    processes = []
    for i, (ckpt, device) in enumerate(zip(args.ckpts, args.devices)):
        print(f"Starting evaluation for checkpoint {i+1} on GPU {device}")
        p = launch_eval_in_process(ckpt, device, args.model,args.ensembler)
        processes.append(p)