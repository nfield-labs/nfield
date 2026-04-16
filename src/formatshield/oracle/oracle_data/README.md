# oracle_data

Pre-trained ThresholdOracle models.

## threshold_oracle_v1.pkl

Trained from benchmark data via:
```bash
uv run formatshield benchmark --backends groq --tasks gsm medical_ner template_fill --output ./results/
python -c "
from formatshield.oracle import ThresholdOracle
oracle = ThresholdOracle.from_benchmark_data('results/benchmark_results.csv')
oracle.save('src/formatshield/oracle/oracle_data/threshold_oracle_v1.pkl')
print('Validation accuracy:', oracle.validation_accuracy)
"
```

When no pkl is present, the oracle uses heuristic thresholds (see threshold_oracle.py).
