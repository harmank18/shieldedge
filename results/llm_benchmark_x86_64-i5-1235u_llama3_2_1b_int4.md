# Llama 3.2 1B-Instruct on 12th Gen Intel(R) Core(TM) i5-1235U

Program `llama3_2_1b_int4.pte` (765 MB), ExecuTorch `llama_main` + XNNPACK, greedy decoding, up to 96 new tokens, 10 runs after one warm-up (2026-11-19 00:05, x86_64, 12 CPUs, 15.3 GB RAM).

| Metric | Mean | Std |
|---|---|---|
| model_load_ms | 1892.0 | 74.4 |
| ttft_ms | 1709.3 | 109.0 |
| prefill_tok_s | 113.3 | 2.5 |
| decode_tok_s | 34.0 | 0.5 |
| generation_ms | 3829.8 | 191.4 |
| wall_ms | 5838.0 | 160.6 |
| peak_rss_mb | 1078.9 | 0.1 |
| prompt_tokens | 193.6 | 12.1 |
| generated_tokens | 72.0 | 8.2 |

Explanations passing validation: 100%
