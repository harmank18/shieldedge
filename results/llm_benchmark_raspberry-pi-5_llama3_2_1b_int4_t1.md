# Llama 3.2 1B-Instruct on Raspberry Pi 5 Model B Rev 1.0

Program `llama3_2_1b_int4.pte` (765 MB), ExecuTorch `llama_main` + XNNPACK, greedy decoding, CPU threads: 1, up to 96 new tokens, 10 runs after one warm-up (2026-11-19 11:31, aarch64, 4 CPUs, 7.8 GB RAM).

| Metric | Mean | Std |
|---|---|---|
| model_load_ms | 2680.7 | 25.0 |
| ttft_ms | 4515.1 | 291.1 |
| prefill_tok_s | 42.9 | 0.1 |
| decode_tok_s | 15.4 | 0.3 |
| generation_ms | 9484.9 | 724.9 |
| wall_ms | 12406.5 | 733.0 |
| peak_rss_mb | 1075.3 | 1.4 |
| prompt_tokens | 193.6 | 12.1 |
| generated_tokens | 76.4 | 11.7 |

Explanations passing validation: 100%
