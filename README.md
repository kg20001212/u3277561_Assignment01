# u3277561_Assignment01

How to run:
1) Install deps: `py -m pip install -r requirements.txt`
2) Open and run `assignment.ipynb` (Parts A & B). Outputs go to `reports/`.
3) Reproduce pipeline: `dvc repro`.

Results (from the notebook):
- Regression MSE: sklearn ~0.000487, manual GD ~0.028640
- Classification F1: Logistic & DecisionTree = 1.0; SGD ≈ 0.999; KNN ≈ 0.996
