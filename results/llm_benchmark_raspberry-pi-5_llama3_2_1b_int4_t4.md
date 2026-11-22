# Llama 3.2 1B-Instruct on Raspberry Pi 5 Model B Rev 1.0

Program `llama3_2_1b_int4.pte` (765 MB), ExecuTorch `llama_main` + XNNPACK, greedy decoding, CPU threads: 4, up to 96 new tokens, 10 runs after one warm-up (2026-11-19 11:26, aarch64, 4 CPUs, 7.8 GB RAM).

| Metric | Mean | Std |
|---|---|---|
| model_load_ms | 2685.8 | 16.9 |
| ttft_ms | 1679.7 | 157.2 |
| prefill_tok_s | 115.6 | 5.6 |
| decode_tok_s | 11.9 | 0.1 |
| generation_ms | 8087.9 | 911.0 |
| wall_ms | 11014.8 | 909.9 |
| peak_rss_mb | 1075.2 | 1.3 |
| prompt_tokens | 193.6 | 12.1 |
| generated_tokens | 76.4 | 11.7 |

Explanations passing validation: 100%
