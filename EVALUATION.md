# Golden Dataset — Hallucination RAG-Pipeline Evaluation

## Overview
30 manually crafted test cases using NVIDIA Q4 FY2024 earnings report.
Each test case has a question, a deliberately correct or hallucinated answer,
the expected verdict, and the actual verdict after running through the system.

---

## Test Cases

| # | Question | Answer | Hallucination Type | Expected Verdict |
|---|---|---|---|---|
| 1 | What was NVIDIA's revenue in Q4 FY2024? | NVIDIA's Q4 FY2024 revenue was $22.1 billion. | None (correct) | GROUNDED |
| 2 | What was NVIDIA's revenue in Q4 FY2024? | NVIDIA's Q4 FY2024 revenue was $99 billion. | Numerical | HALLUCINATION |
| 3 | What was NVIDIA's revenue in Q4 FY2024? | NVIDIA reported a net loss in Q4 FY2024 and revenue declined significantly. | Contradiction | HALLUCINATION |
| 4 | What was NVIDIA's gross margin in Q4 FY2024? | NVIDIA's gross margin in Q4 FY2024 was 76.0%. | None (correct) | GROUNDED |
| 5 | What was NVIDIA's gross margin in Q4 FY2024? | NVIDIA's gross margin in Q4 FY2024 was 54.0%. | Numerical | HALLUCINATION |
| 6 | What was NVIDIA's Data Center revenue in Q4 FY2024? | Data Center revenue was $18.4 billion in Q4 FY2024. | None (correct) | GROUNDED |
| 7 | What was NVIDIA's Data Center revenue in Q4 FY2024? | Data Center revenue was $5.2 billion in Q4 FY2024. | Numerical | HALLUCINATION |
| 8 | Who is the CEO of NVIDIA? | Jensen Huang is the founder and CEO of NVIDIA. | None (correct) | GROUNDED |
| 9 | Who is the CEO of NVIDIA? | Elon Musk is the CEO of NVIDIA. | Factual | HALLUCINATION |
| 10 | What was NVIDIA's full year revenue for FY2024? | NVIDIA's full year revenue for FY2024 was $60.9 billion. | None (correct) | GROUNDED |
| 11 | What was NVIDIA's full year revenue for FY2024? | NVIDIA's full year revenue for FY2024 was $30 billion. | Numerical | HALLUCINATION |
| 12 | What was the Q4 FY2024 GAAP earnings per diluted share? | GAAP earnings per diluted share was $4.93 in Q4 FY2024. | None (correct) | GROUNDED |
| 13 | What was the Q4 FY2024 GAAP earnings per diluted share? | GAAP earnings per diluted share was $1.20 in Q4 FY2024. | Numerical | HALLUCINATION |
| 14 | How much did NVIDIA's revenue grow year over year in Q4 FY2024? | Revenue grew 265% year over year in Q4 FY2024. | None (correct) | GROUNDED |
| 15 | How much did NVIDIA's revenue grow year over year in Q4 FY2024? | Revenue declined 10% year over year in Q4 FY2024. | Contradiction | HALLUCINATION |
| 16 | What was NVIDIA's Gaming revenue in Q4 FY2024? | Gaming revenue was $2.9 billion in Q4 FY2024. | None (correct) | GROUNDED |
| 17 | What was NVIDIA's Gaming revenue in Q4 FY2024? | Gaming revenue was $8.5 billion in Q4 FY2024. | Numerical | HALLUCINATION |
| 18 | Who is the CFO of NVIDIA? | Colette Kress is NVIDIA's executive vice president and CFO. | None (correct) | GROUNDED |
| 19 | Who is the CFO of NVIDIA? | Jensen Huang is both the CEO and CFO of NVIDIA. | Factual | HALLUCINATION |
| 20 | What is NVIDIA's expected revenue for Q1 FY2025? | NVIDIA expects revenue of $24.0 billion for Q1 FY2025. | None (correct) | GROUNDED |
| 21 | What is NVIDIA's expected revenue for Q1 FY2025? | NVIDIA expects revenue of $10.0 billion for Q1 FY2025. | Numerical | HALLUCINATION |
| 22 | What was NVIDIA's net income in Q4 FY2024? | NVIDIA's net income in Q4 FY2024 was $12.285 billion. | None (correct) | GROUNDED |
| 23 | What was NVIDIA's net income in Q4 FY2024? | NVIDIA reported a net loss of $2 billion in Q4 FY2024. | Contradiction | HALLUCINATION |
| 24 | What was NVIDIA's Automotive revenue in Q4 FY2024? | Automotive revenue was $281 million in Q4 FY2024. | None (correct) | GROUNDED |
| 25 | What was NVIDIA's Automotive revenue in Q4 FY2024? | Automotive revenue was $2.8 billion in Q4 FY2024. | Numerical | HALLUCINATION |
| 26 | When was NVIDIA founded? | NVIDIA was founded in 1993. | None (correct) | GROUNDED |
| 27 | When was NVIDIA founded? | NVIDIA was founded in 2005 by Jensen Huang. | Numerical + Factual | HALLUCINATION |
| 28 | What was the gross margin improvement from Q4 FY23 to Q4 FY24? | Gross margin improved by 12.7 percentage points from Q4 FY23 to Q4 FY24. | None (correct) | GROUNDED |
| 29 | What was the gross margin improvement from Q4 FY23 to Q4 FY24? | Gross margin declined by 5 percentage points from Q4 FY23 to Q4 FY24. | Contradiction | HALLUCINATION |
| 30 | What is NVIDIA's stock ticker symbol? | NVIDIA trades on NASDAQ under the ticker symbol NVDA. | None (correct) | GROUNDED |

---

## Results Summary

| Category | Count |
|---|---|
| Total test cases | 30 |
| Correct answers (should be GROUNDED) | 15 |
| Hallucinated answers (should be HALLUCINATION) | 15 |
| Numerical hallucinations | 10 |
| Contradiction hallucinations | 4 |
| Factual hallucinations | 2 |
| Mixed hallucinations | 1 |

---

## Accuracy Results

> Fill in the "Actual Verdict" column after running each test case through the system.
> Then calculate recall and precision.

| Metric | Formula | Result |
|---|---|---|
| Recall | Hallucinations correctly caught / Total hallucinations | X / 15 |
| Precision | Correct GROUNDEDs / Total predicted GROUNDED | X / 15 |
| Overall Accuracy | Correct verdicts / Total test cases | X / 30 |

---

## How to Run Tests

1. Start the system: `python main.py`
2. Load `data/sample.pdf`
3. For each test case, ask the question
4. After getting the answer, temporarily replace `answer = llm.invoke(prompt)` with the test answer
5. Record the actual verdict
6. Calculate final accuracy

---

## Notes

- Numerical hallucinations are the most reliably caught (expected 80-90% recall)
- Contradiction hallucinations caught by DeBERTa with high confidence (0.99+ scores)
- Factual hallucinations on proper nouns (names, places) are the hardest to catch
- Short bare answers skip contradiction checking by design