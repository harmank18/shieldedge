# Llama 3.2 1B-Instruct on 12th Gen Intel(R) Core(TM) i5-1235U

Program `llama3_2_1b_int4.pte` (765 MB), ExecuTorch `llama_main` + XNNPACK, greedy decoding, up to 96 new tokens, 10 runs after one warm-up (2026-11-18 23:22, x86_64, 12 CPUs, 15.3 GB RAM).

| Metric | Mean | Std |
|---|---|---|
| model_load_ms | 1893.5 | 101.6 |
| ttft_ms | 3036.1 | 130.7 |
| prefill_tok_s | 112.2 | 1.6 |
| decode_tok_s | 31.2 | 0.6 |
| generation_ms | 5489.4 | 330.1 |
| wall_ms | 7500.8 | 389.3 |
| peak_rss_mb | 1078.8 | 0.1 |
| prompt_tokens | 340.4 | 12.2 |
| generated_tokens | 76.4 | 10.5 |

Explanations passing validation: 100%
