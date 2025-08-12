import os
import subprocess
import argparse
from pathlib import Path
from multiprocessing import Process
from datetime import datetime
import time

TASKS = ["bridge.sh"]
PREDEFINED_CKPTS = []

def get_checkpoints(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    return [str(p) for p in sorted(ckpt_dir.iterdir()) if p.is_dir()]

def run_task(ckpt, device, model, ensembler):
    name = Path(ckpt).name
    log_dir = f"results/{name}_{ensembler}"
    os.makedirs(log_dir, exist_ok=True)

    for task in TASKS:
        log = os.path.join(log_dir, f"{name}--{task}.log")
        threshold = "0.5"
        with open(log, "w") as fout, open(log + ".err", "w") as ferr:
            cmd = ["bash", f"scripts/{task}", ckpt, model, "-0.8", log_dir, str(device), ensembler,threshold]
            subprocess.run(cmd, stdout=fout, stderr=ferr)

    with open(f"{log_dir}/total.metrics", "a") as f:
        subprocess.run(["python", "tools/calc_metrics_evaluation_videos.py", "--log-dir-root", log_dir], stdout=f)

def launch_eval(ckpt, device, model, ensembler):
    p = Process(target=run_task, args=(ckpt, device, model, ensembler))
    p.start()
    print(f"Starting evaluation for {ckpt} on GPU {device}")
    return p

def smart_schedule(ckpts, devices, model, ensembler):
    pending = list(enumerate(ckpts))
    running = {}
    completed = []

    for d in devices:
        if pending:
            i, ckpt = pending.pop(0)
            running[d] = (launch_eval(ckpt, d, model, ensembler), i, ckpt)

    while running or pending:
        done = []
        for d, (p, i, ckpt) in running.items():
            if not p.is_alive():
                p.join()
                done.append(d)
                completed.append((i, ckpt))

        for d in done:
            del running[d]
            if pending:
                i, ckpt = pending.pop(0)
                running[d] = (launch_eval(ckpt, d, model, ensembler), i, ckpt)

        if running:
            time.sleep(2)

    return completed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", type=str)
    parser.add_argument("--ckpts", nargs="*")
    parser.add_argument("--devices", type=int, nargs="+")
    parser.add_argument("--model", type=str, default="lisa")
    parser.add_argument("--ensembler", type=str, default="vote")
    parser.add_argument("--use-predefined", action="store_true")
    parser.add_argument("--list-ckpts", action="store_true")
    args = parser.parse_args()

    if args.list_ckpts:
        print("\n".join(PREDEFINED_CKPTS))
        exit(0)

    if args.ckpts:
        ckpts = args.ckpts
    elif args.use_predefined:
        ckpts = PREDEFINED_CKPTS
    else:
        ckpts = get_checkpoints(args.ckpt_dir)

    ckpts = [ckpt for ckpt in ckpts if os.path.exists(ckpt)]
    if not ckpts:
        print("No valid checkpoints found.")
        exit(1)

    devices = args.devices or [0]
    print(f"Model: {args.model}, Ensembler: {args.ensembler}, Devices: {devices}, Checkpoints: {len(ckpts)}")

    t0 = datetime.now()
    completed = smart_schedule(ckpts, devices, args.model, args.ensembler)
    t1 = datetime.now()

    print(f"Done: {len(completed)} checkpoints")
    print(f"Time: {t0.strftime('%F %T')} → {t1.strftime('%F %T')} ({t1 - t0})")