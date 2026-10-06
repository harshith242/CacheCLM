# Extra sample: a second EventQA book, editing arms only

Config: `configs/deepseek_extra.yaml`. Book 2 is EventQA row 13 (`eventqa_131072`), characters 285K-450K: 41K tokens in 25 parts, about 4x the 10K budget. It has 30 questions, the ones whose events fall in that stretch. DeepSeek Flash with thinking off; one run per arm. The summary, none and full arms were not run, so the standard report, which needs all four primary arms, was not produced.

| Arm | Accuracy | Billed $ | Calls | Cache hit rate | Output tokens | Edits applied | Rejected (price) | Overflow stop |
|---|---|---|---|---|---|---|---|---|
| clm | 0.97 | 0.147 | 87 | 63% | 85,535 | 36 | — | none |
| gate | 0.90 | 0.158 | 90 | 67% | 96,486 | 20 | 17 | none |
| skill | 0.87 | 0.081 | 57 | 80% | 45,077 | 15 | — | part 10 (13 of 25 unread) |

## Where the money went (DeepSeek prices, both books)

| Book | Arm | Total $ | Cache hits | Cache misses | Output | Output share |
|---|---|---|---|---|---|---|
| 1 | summary | 0.071 | 0.001 | 0.040 | 0.030 | 42% |
| 1 | clm | 0.176 | 0.001 | 0.044 | 0.130 | 74% |
| 1 | gate | 0.098 | 0.002 | 0.025 | 0.071 | 73% |
| 1 | skill | 0.245 | 0.003 | 0.055 | 0.187 | 76% |
| 2 | clm | 0.147 | 0.001 | 0.042 | 0.103 | 70% |
| 2 | gate | 0.158 | 0.002 | 0.040 | 0.116 | 73% |
| 2 | skill | 0.081 | 0.002 | 0.025 | 0.054 | 67% |

Spend: $0.38.
