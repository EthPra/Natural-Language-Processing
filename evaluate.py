import json
import string
import re
import unicodedata
from collections import Counter

def normalise(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation
                and not unicodedata.category(ch).startswith('P')
                and unicodedata.category(ch) != 'Sc')
    return ' '.join(s.split())

with open("data/QA_pairs.json", encoding="utf-8") as f:
    qa_pairs = json.load(f)

with open("predictions.json", encoding="utf-8") as f:
    preds = {p["question_id"]: p["answer"] for p in json.load(f)}

em_count = 0
f1_total = 0

for item in qa_pairs:
    gold = normalise(item["gold_answer"])
    pred = normalise(preds.get(item["question_id"], ""))

    # exact match
    em = gold == pred
    em_count += em

    # f1
    gold_toks = gold.split()
    pred_toks = pred.split()
    common = sum((Counter(pred_toks) & Counter(gold_toks)).values())
    if common:
        prec = common / len(pred_toks)
        rec = common / len(gold_toks)
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0
    f1_total += f1

    if not em:
        print(f"[miss] {item['question_id']}")
        print(f"  q: {item['question']}")
        print(f"  gold: {item['gold_answer']}")
        print(f"  pred: {preds.get(item['question_id'], '')}")
        print(f"  f1: {f1:.2f}\n")

n = len(qa_pairs)
print(f"EM: {100 * em_count / n:.2f}%")
print(f"F1: {100 * f1_total / n:.2f}%")