# cs-175

Compare **random monkeys** vs a **bigram-trained monkey** (trained on Shakespeare) and score both with the current judge:

```bash
python3 expirements/compare_bigram_vs_random.py
# or:
python3 -m expirements.compare_bigram_vs_random
```

Run the **Old English start → Shakespeare push** experiment (bigram updates guided by the judge):

```bash
# Top-K updates (baseline)
python3 expirements/old_english_start_bigram_push.py --mode topk --decay 0.99

# Score-weighted updates (smoother)
python3 expirements/old_english_start_bigram_push.py --mode weighted --decay 0.99 --weight-scale 200

# Optional anti-copy filter
python3 expirements/old_english_start_bigram_push.py --mode topk --anti-copy
```
