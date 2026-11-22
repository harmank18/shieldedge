# Curated India-specific evaluation sets

These small sets complement the public benchmarks, which contain very little
Indian content. They are used to report a separate "India challenge set" score.
**They are not public benchmarks, and the numbers from them are indicative only.**

| File | Rows | What it is |
|---|---|---|
| `india_urls.csv` | 65 | `real_site`: real Indian websites that are *not* on ShieldEdge's official-domain allowlist (tests generalisation). `official_allowlisted`: real official bank/government URLs (these pass through the allowlist, so they only test the allowlist). `team_authored`: **synthetic** phishing URLs written by the team to mimic patterns from CERT-In advisories and news reports. They are not real captured phishing URLs. **Evaluation only, never used for training.** |
| `india_sms_train.csv` | see file | **Team-authored** Indian transactional (ham) and scam SMS used for training. |
| `india_sms_test.csv` | see file | **Team-authored** Indian SMS written separately from the training file (different templates, senders and amounts). **Evaluation only.** |

Authors: the ShieldEdge team (Harman Kishnadwala, Arnav Mohan Jabade, Aishan Prabahakar Nikam).
