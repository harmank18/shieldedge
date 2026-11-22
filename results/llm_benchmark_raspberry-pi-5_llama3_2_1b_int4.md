# Llama 3.2 1B-Instruct on Raspberry Pi 5 Model B Rev 1.0

Program `llama3_2_1b_int4.pte` (765 MB), ExecuTorch `llama_main` + XNNPACK, greedy decoding, CPU threads: 2, up to 96 new tokens, 10 runs after one warm-up (2026-11-19 23:04, aarch64, 4 CPUs, 7.8 GB RAM).

| Metric | Mean | Std |
|---|---|---|
| model_load_ms | 2675.4 | 2.9 |
| ttft_ms | 2536.1 | 186.0 |
| prefill_tok_s | 76.4 | 1.4 |
| decode_tok_s | 14.5 | 0.5 |
| generation_ms | 7823.8 | 719.2 |
| wall_ms | 10741.5 | 716.5 |
| peak_rss_mb | 1075.7 | 1.1 |
| prompt_tokens | 193.6 | 12.1 |
| generated_tokens | 76.4 | 11.7 |

Explanations passing validation: 100%
