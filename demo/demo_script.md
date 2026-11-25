# ShieldEdge demo video: 2-minute storyboard

This is the storyboard of the submitted 2-minute demo video (1920×1080, MP4). Everything was recorded on the
**Raspberry Pi 5** with no internet connection, driven over SSH from a laptop. Llama explanations play at
real speed. Run these commands to reproduce each shot.

## Setup

```bash
cd ~/shieldedge && source venv/bin/activate
python shieldedge.py --clear-logs          # start with an empty audit log
python shieldedge.py --web                 # dashboard (separate terminal), served on 127.0.0.1:8080
# from the laptop: ssh -fN -L 8081:localhost:8080 <pi>   then open http://localhost:8081
```

To take the Pi offline without losing the SSH session, remove its default route
(`sudo ip route del default`) instead of turning Wi-Fi off. Run every command once before recording, so the
first Llama run is not slowed down by a cold disk.

## Shots

| Time | Shot | Command or action | What it shows |
|---|---|---|---|
| 0:00–0:12 | Intro | dashboard at http://localhost:8081 | team, PS3, the idea |
| 0:12–0:19 | Device, offline | `cat /proc/device-tree/model` and `ping -c 2 8.8.8.8` | "Raspberry Pi 5 Model B", then "Network is unreachable" |
| 0:19–0:41 | Phishing link + Llama | `python shieldedge.py --url "http://sbi-net-banking-update.xyz/login"` with htop alongside | PHISHING with reasons; Llama 3.2 1B (INT4, ExecuTorch) streams an explanation while two cores are busy; vetted safety steps (1930, cybercrime.gov.in) |
| 0:41–0:51 | Hinglish scam SMS | `python shieldedge.py --sms "Aapka SBI khata band ho jayega. Turant KYC update karo: http://sbi-kyc-help.xyz"` | SCAM, Hinglish pressure phrases, fake SBI link, Llama explanation |
| 0:51–0:56 | Genuine bank SMS | `python shieldedge.py --sms "Rs.2,500.00 debited from A/c XX4521 to VPA zomato@icici. Not you? Call 18002586161 -HDFC Bank" --fast` | SAFE (no false alarm) |
| 0:56–1:02 | QR code | `python shieldedge.py --qr data/samples/demo_qr.png --fast` | decodes `paytm-kyc-pending.shop`, PHISHING |
| 1:02–1:27 | Network attack | dashboard during the DDoS phase of the CIC-IDS-2017 replay; click a flagged flow, then **Explain with Llama** | red DDoS block in the traffic chart, dataset label agrees, Llama explanation, per-core CPU load |
| 1:27–1:37 | Proof nothing leaves the device | `ss -tunap \| grep -E "python\|llama_main" \|\| echo "no network sockets"` and `tail -4 results/raspberry-pi-5/offline_proof.txt` | no sockets; 0 packets captured during a Llama run |
| 1:37–1:52 | Tests and results | `python -m pytest -q`, then `results/benchmarks.md` on GitHub | 45 passed; detection quality table; Raspberry Pi Llama speed table |
| 1:52–1:58 | Close | the repository README | open source, reproducible |

## Narration (about 2.5 words per second)

**0:00–0:41.**
"Hi, we're team ShieldEdge from BITS Pilani Goa: Harman, Arnav and Aishan, mentored by Professor Akshay Madan.
For Problem Statement 3, we built an offline detector for phishing, scam SMS and network attacks on a Raspberry
Pi. This is our Raspberry Pi 5, with no internet at all. Everything runs locally on its Arm CPU. A fake SBI link
is flagged as phishing in milliseconds. Now watch the CPU: that's Llama 3.2 1B, quantised to 4-bit and running
through ExecuTorch, explaining the threat live on the Pi. Llama only explains. The safety steps, like calling
1930, come from a fixed, checked list."

**0:41–1:27.**
"Scammers write in Hinglish too. This fake KYC message is caught as a scam, with the exact pressure phrases and
the fake SBI link listed as reasons. A real HDFC debit alert stays safe, so users can keep trusting the
warnings. QR codes are checked too. This one hides a fake Paytm KYC link. This dashboard runs on the Pi. It
replays real CIC-IDS-2017 network traffic, and right now a DDoS flood shows up in red. Our network model was
trained only on normal traffic, yet on the test day it caught every DDoS and botnet flow, with an F1 of 0.98.
Here Llama explains a flagged flow, again fully on the device."

**1:27–1:58.**
"To prove it's private: ShieldEdge holds no network connections, and a packet capture during a Llama run
recorded zero packets. All 45 tests pass. On unseen data, we reach an F1 of 0.94 for links and 0.96 for real
scam SMS. With 4-bit quantisation and Arm KleidiAI kernels, a full explanation takes about eight seconds on the
Pi, using one gigabyte of memory. Everything is open source on GitHub. That's ShieldEdge. Thank you!"
