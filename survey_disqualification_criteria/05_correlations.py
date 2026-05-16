"""
05_correlations.py
===================
Section 5.3 of the Stan_prac.pdf document: Correlation of experience with rating severity.

Examines whether a respondent's length of experience with maps (and their self-assessed expertise)
correlates with the mean rating they assign to criteria — interpreted as
"severity" of judgment.

Uses Spearman's rank correlation coefficient (rho), appropriate for ordinal data.

Run:
    python 05_correlations.py
"""
import numpy as np
from scipy import stats
from data_loader import (
    load_ratings, parse_years,
    COL_YEARS_MAPS, COL_YEARS_STAT, COL_SELF_EXPERT,
)


def main():
    ratings, df = load_ratings()
    N = len(df)

    # Respondent severity = mean of all their ratings (ignores NaNs)
    respondent_mean = ratings.mean(axis=1)

    # Years of experience with maps (after converting text values to numbers)
    years_maps = np.array([parse_years(v) for v in df.iloc[:, COL_YEARS_MAPS]],
                          dtype=object)
    years_stat = np.array([parse_years(v) for v in df.iloc[:, COL_YEARS_STAT]],
                          dtype=object)
    self_expert = df.iloc[:, COL_SELF_EXPERT].values

    def spearman(x_raw, y_raw, label):
        # Filter out None / NaN pairs
        pairs = [(x, y) for x, y in zip(x_raw, y_raw)
                 if x is not None and not (isinstance(x, float) and np.isnan(x))
                 and y is not None and not (isinstance(y, float) and np.isnan(y))]
        if len(pairs) < 3:
            print(f"  {label}: too few data points (N = {len(pairs)})")
            return
        xs, ys = zip(*pairs)
        rho, p = stats.spearmanr(xs, ys)
        sig = "" if p >= 0.05 else "  <- significant (p < 0.05)"
        print(f"  {label:<55} N = {len(pairs):>2}  rho = {rho:>7.4f}  p = {p:>6.4f}{sig}")

    print("="*80)
    print("Spearman correlation between respondent characteristics and rating 'severity'")
    print("(Severity = mean of all 10 ratings assigned by the respondent)")
    print("="*80)
    print()

    spearman(years_maps,  respondent_mean, "Years of experience with maps (overall)")
    spearman(years_stat,  respondent_mean, "Years of experience with statistical maps")
    spearman(self_expert, respondent_mean, "Self-assessed expertise (1-5)")

    print("\nInterpretation:")
    print("  Negative rho — more experienced experts are LESS severe")
    print("  Positive rho — more experienced experts are MORE severe")
    print("  Significance: p < 0.05 means the observed correlation is unlikely")
    print("                under the null hypothesis of no association.")


if __name__ == "__main__":
    main()
