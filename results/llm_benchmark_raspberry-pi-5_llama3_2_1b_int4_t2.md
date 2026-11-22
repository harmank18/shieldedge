# Llama 3.2 1B-Instruct on Raspberry Pi 5 Model B Rev 1.0

Program `llama3_2_1b_int4.pte` (765 MB), ExecuTorch `llama_main` + XNNPACK, greedy decoding, CPU threads: 2, up to 96 new tokens, 10 runs after one warm-up (2026-11-19 11:28, aarch64, 4 CPUs, 7.8 GB RAM).

| Metric | Mean | Std |
|---|---|---|
| model_load_ms | 2680.9 | 18.5 |
| ttft_ms | 2487.0 | 168.2 |
| prefill_tok_s | 77.9 | 0.8 |
| decode_tok_s | 14.9 | 0.5 |
| generation_ms | 7598.4 | 681.8 |
| wall_ms | 10518.3 | 669.9 |
| peak_rss_mb | 1075.5 | 1.4 |
| prompt_tokens | 193.6 | 12.1 |
| generated_tokens | 76.4 | 11.7 |

Explanations passing validation: 100%
