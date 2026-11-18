# Llama 3.2 1B-Instruct on 12th Gen Intel(R) Core(TM) i5-1235U

Program `llama3_2_1b.pte` (2358 MB), ExecuTorch `llama_main` + XNNPACK, greedy decoding, up to 96 new tokens, 3 runs after one warm-up (2026-11-18 23:39, x86_64, 12 CPUs, 15.3 GB RAM).

| Metric | Mean | Std |
|---|---|---|
| model_load_ms | 44.3 | 0.6 |
| ttft_ms | 190726.3 | 18223.0 |
| prefill_tok_s | 1.8 | 0.1 |
| decode_tok_s | 1.3 | 0.0 |
| generation_ms | 264253.0 | 18573.5 |
| wall_ms | 264400.2 | 18586.7 |
| peak_rss_mb | 2515.9 | 0.1 |
| prompt_tokens | 343.0 | 17.5 |
| generated_tokens | 95.0 | 0.0 |

Explanations passing validation: 0%
