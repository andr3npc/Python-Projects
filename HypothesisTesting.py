# Import packages
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import pingouin
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
drug_safety = pd.read_csv("drug_safety.csv")
print(drug_safety.head())
print(f"\nShape: {drug_safety.shape}")
print(f"\nColumns: {list(drug_safety.columns)}")
print(f"\nTreatment groups:\n{drug_safety['trx'].value_counts()}")

# ============================================================
# 1. Two-sample z-test for proportions of adverse effects
#    H0: proportion of adverse effects is the same in Drug and Placebo groups
# ============================================================

# Group counts
drug_group = drug_safety[drug_safety["trx"] == "Drug"]
placebo_group = drug_safety[drug_safety["trx"] == "Placebo"]

# Number with adverse effects in each group
# adverse_effects column contains "Yes"/"No" strings, so count "Yes" occurrences
n_adverse_drug = (drug_group["adverse_effects"] == "Yes").sum()
n_adverse_placebo = (placebo_group["adverse_effects"] == "Yes").sum()

# Total in each group
n_drug = len(drug_group)
n_placebo = len(placebo_group)

# Two-sample z-test for proportions
count = np.array([n_adverse_drug, n_adverse_placebo])
nobs = np.array([n_drug, n_placebo])

z_stat, two_sample_p_value = proportions_ztest(count, nobs)

print(f"\n--- Two-sample proportions z-test ---")
print(f"Z-statistic: {z_stat:.4f}")
print(f"two_sample_p_value: {two_sample_p_value:.4f}")

# ============================================================
# 2. Chi-square test of independence: num_effects vs trx
#    H0: number of adverse effects is independent of treatment group
# ============================================================

# Create a contingency table and run chi-square test using pingouin
num_effects_groups = pd.crosstab(drug_safety["num_effects"], drug_safety["trx"])
expected, observed, stats = pingouin.chi2_independence(
    drug_safety, x="num_effects", y="trx"
)

# Extract the p-value from the Pearson chi-square row
num_effects_p_value = stats[stats["test"] == "pearson"]["pval"].values[0]

print(f"\n--- Chi-square test of independence (num_effects vs trx) ---")
print(f"num_effects_p_value: {num_effects_p_value:.4f}")

# ============================================================
# 3. Mann-Whitney U test: age between Drug and Placebo groups
#    H0: no significant difference in age between the two groups
# ============================================================

age_test = pingouin.mwu(
    drug_group["age"],
    placebo_group["age"],
    alternative="two-sided",
)

age_group_effects_p_value = age_test["p-val"].values[0]

print(f"\n--- Mann-Whitney U test (age: Drug vs Placebo) ---")
print(f"age_group_effects_p_value: {age_group_effects_p_value:.4f}")

# ============================================================
# Summary
# ============================================================
print("\n========== SUMMARY ==========")
print(f"two_sample_p_value:          {two_sample_p_value}")
print(f"num_effects_p_value:         {num_effects_p_value}")
print(f"age_group_effects_p_value:   {age_group_effects_p_value}")
