import numpy as np
import pandas as pd
from poisonshield.evaluation.statistics import summarize, paired_tests


def test_summary_and_holm():
    rows=[]
    for seed in range(5):
        for method, score in [("PoisonShieldAI", 0.9), ("poisoned", 0.7), ("DBSCAN_only", 0.8)]:
            rows.append({"dataset":"synthetic", "task":"binary", "victim":"rf", "poison_rate":0.1,
                         "method":method, "seed":seed, "status":"ok", "macro_f1":score+0.001*seed,
                         "accuracy":score, "weighted_f1":score, "clean_retention":0.95,
                         "poison_detection_recall":0.7})
    df=pd.DataFrame(rows)
    summary=summarize(df)
    assert (summary.n == 5).all()
    assert not summary.ci95_high.isna().any()
    tests=paired_tests(df)
    assert len(tests) == 2
    assert tests.p_holm.between(0,1).all()
