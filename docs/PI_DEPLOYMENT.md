# Running ShieldEdge on a Raspberry Pi 5

The goal is clone, run one script, record. Everything heavy (training and the
Llama export) happens on the laptop. The Pi only builds the small C++ runner.

## 0. OS

Use **64-bit** Raspberry Pi OS (Bookworm or newer) or Debian 12+ arm64. Both are
Debian, so either works and there is no need to install Ubuntu. Check with
`uname -m`, which must print `aarch64`. A Pi 5 with 4 GB RAM is enough (Llama
peaks at about 1.1 GB on the laptop); 8 GB gives more headroom. Use the active
cooler or a heatsink, because the build and the benchmarks load all 4 cores.

## 1. Clone the repo on the Pi

```bash
git clone -b harman-devs https://github.com/harmank18/shieldedge.git ~/shieldedge
cd ~/shieldedge
```

## 2. Copy the Llama files from the laptop

These are too big for git (`.pte` is 766 MB). Run this **on the laptop**:

```bash
./scripts/copy_to_pi.sh <user>@<pi-address>
```

This copies `executorch/llama3_2_1b_int4.pte` and
`models/Llama-3.2-1B-Instruct/original/tokenizer.model`. The `.pte` is
architecture-independent: the laptop export runs unchanged on the Pi.

## 3. Set up and build the runner on the Pi

```bash
./scripts/setup_pi.sh
```

This installs the apt packages, creates the venv, builds ExecuTorch v1.6.0's
`llama_main` with XNNPACK (Arm KleidiAI kernels enabled by default) using
`make llama-cpu`, and copies `llama_main` and `libllama_runner.so` into
`executorch/`. The build takes roughly 30–60 minutes. If it stops on missing Python modules,
run `cd ~/executorch && ./install_requirements.sh` and then re-run the script.

## 4. Smoke test

```bash
source venv/bin/activate
python shieldedge.py --url "http://sbi-net-banking-update.xyz/login"
python shieldedge.py --sms "Aapka SBI khata band ho jayega. Turant KYC update karo: http://sbi-kyc-help.xyz"
python shieldedge.py --flows data/samples/demo_flows.csv
```

## 5. Record the numbers for the report

```bash
./scripts/benchmark_device.sh                 # latency, Llama benchmark (10 runs), prompt eval, thermals
nmcli radio wifi off                          # or unplug Ethernet
./scripts/offline_proof.sh                    # ping fails, tcpdump shows 0 packets during inference
nmcli radio wifi on
git add results/ && git commit -m "Add Raspberry Pi 5 benchmark results"
```

Detection *quality* (precision, recall and so on) depends only on the models,
which are identical on both machines, so it is measured once on the laptop.
The Pi run adds per-input latency, Llama speed and memory, thermals, and the
offline proof. These are labelled `raspberry-pi-5` automatically.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `llama_main: error while loading shared libraries: libllama_runner.so` | Copy the `.so` next to `llama_main` (setup does this). The app also sets `LD_LIBRARY_PATH=executorch/`. |
| Explanation falls back to the template | Check that `executorch/llama3_2_1b_int4.pte` and the tokenizer exist; run `executorch/llama_main --help`. |
| Slow or throttled | `vcgencmd get_throttled` should print `0x0`. Improve cooling, or set `SHIELDEDGE_LLM_THREADS=4`. |
| Llama takes too long for the demo | `SHIELDEDGE_LLM_MAX_TOKENS=64` shortens explanations. `--fast` skips Llama entirely. |
