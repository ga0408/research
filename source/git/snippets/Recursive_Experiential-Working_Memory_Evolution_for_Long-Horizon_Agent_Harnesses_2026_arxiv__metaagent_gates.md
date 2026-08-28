# Recuris Meta-Agent Discipline & Admission Gates — 코드 스니펫

> 출처: `src/recuris/metaagent/gates.py` / [분석 문서](../../../report/[paper][git]_Recursive_Experiential-Working_Memory_Evolution_for_Long-Horizon_Agent_Harnesses_2026_arxiv.md)

## 1. Paired Held-Out Significance Gate (`held_out_paired_gate`)

```python
def held_out_paired_gate(base, cand, alpha=0.05, reg_cap=0, eps=1e-9,
                         n_boot=3000, seed=0, material=0.0):
    """Paired held-out significance test. This is the only admission gate.
    
    Per item, take the mean over seeds, then the difference d_i between the
    candidate and the base. Bootstrap over ITEMS, not over trials: trials
    within an item are not independent.
    
    ACCEPT iff the interval excludes zero on the improving side (lo > 0)
    and the number of regressed items is at most reg_cap.
    """
    items = sorted(set(base) & set(cand))
    diffs = [statistics.mean(cand[i]) - statistics.mean(base[i]) for i in items]
    if not diffs:
        return Verdict(False, 0.0, (0.0, 0.0), 0, 0, "no comparable items")
    net = statistics.mean(diffs)
    floor = max(eps, material)
    n_up = sum(1 for x in diffs if x > floor)
    n_dn = sum(1 for x in diffs if x < -floor)
    rng = random.Random(seed)
    boots = sorted(statistics.mean([diffs[rng.randrange(len(diffs))] for _ in diffs])
                   for _ in range(n_boot))
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    accept = (lo > 0) and (n_dn <= reg_cap)
    if accept:
        reason = "net improvement, CI excludes 0"
    elif lo <= 0:
        reason = "CI includes 0: not significant"
    else:
        reason = f"{n_dn} regressed items exceeds the cap of {reg_cap}"
    return Verdict(accept, round(net, 4), (round(lo, 4), round(hi, 4)), n_up, n_dn, reason)
```

## 2. Test-Set Leakage & Fingerprint Verification

```python
def leakage_check(card, test_gold_params):
    """False if the card body contains any test-set answer parameter."""
    body = (card.get("body") or "").lower()
    return not any(str(g).lower() in body for g in test_gold_params)


def fingerprint_verify(cand_results, prescription):
    """False unless the prescribed carrier fired at least once.
    
    Without this, a change that improved the score for an unrelated reason
    cannot be credited to the mechanism it claimed to fix.
    """
    return cand_results.fingerprint.get(prescription.carrier.value, 0) > 0
```
