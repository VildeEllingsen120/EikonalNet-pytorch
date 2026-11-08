#!/usr/bin/env python3
"""Run train.py for several seeds, N at a time (one torch thread each).

usage: python scripts/run_seeds.py <run_prefix> <data_seed> <seeds comma-separated> [--jobs 4] -- [train.py flags]
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        own, flags = argv[:i], argv[i + 1:]
    else:
        own, flags = argv, []
    prefix, ds, seeds = own[0], own[1], [int(s) for s in own[2].split(",")]
    jobs = int(own[own.index("--jobs") + 1]) if "--jobs" in own else 4
    pending = list(seeds)
    running = {}
    t0 = time.time()
    while pending or running:
        while pending and len(running) < jobs:
            s = pending.pop(0)
            run = f"{prefix}_d{ds}_s{s}"
            log = open(os.path.join(ROOT, "results", f"{run}.log"), "w")
            cmd = [sys.executable, os.path.join(HERE, "train.py"), "--run_name", run, "--data_seed", str(ds),
                   "--seed", str(s), "--threads", "1", *flags]
            running[run] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT), log)
        for run, (p, log) in list(running.items()):
            if p.poll() is not None:
                log.write(f"exit {p.returncode}\n"); log.close()
                print(f"done {run} exit {p.returncode} ({time.time() - t0:.0f}s elapsed)", flush=True)
                del running[run]
        time.sleep(2)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
