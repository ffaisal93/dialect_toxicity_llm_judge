import pandas as pd
import scipy.stats as stats

# === Load Excel ===
xls = pd.ExcelFile("human-study/annotation.xlsx")

# === Read Sheets ===
bs1 = xls.parse("annotator_1_bs1")
bs2 = xls.parse("annotator_2_bs2")
es1 = xls.parse("annotator_2_es1")
es2 = xls.parse("annotator_1_es2")

# === Combine Scores ===
bengali_scores = pd.concat([bs1["Rating"], bs2["Rating"]], ignore_index=True)
english_scores = pd.concat([es1["Rating"], es2["Rating"]], ignore_index=True)

# === Summary Stats ===
print("📈 Toxicity Score Summary")
print("-------------------------")
bengali_mean = bengali_scores.mean()
bengali_median = bengali_scores.median()
english_mean = english_scores.mean()
english_median = english_scores.median()
print("Bengali Toxicity: Mean =", bengali_mean, "Median =", bengali_median)
print("English Toxicity: Mean =", english_mean, "Median =", english_median)

# === Score Distribution ===
def get_score_distribution(scores):
    counts = scores.value_counts().sort_index()
    total = len(scores)
    return {score: round((counts.get(score, 0) / total) * 100, 1) for score in range(1, 6)}

bengali_dist = get_score_distribution(bengali_scores)
english_dist = get_score_distribution(english_scores)

# === Mann-Whitney U Test ===
stat, pval = stats.mannwhitneyu(bengali_scores, english_scores, alternative='two-sided')
print("\n📊 Mann-Whitney U Test")
print("------------------------")
print(f"  Statistic = {stat:.3f}")
print(f"  p-value   = {pval:.4f}")

# === LaTeX Table Output ===
print("\n📄 LaTeX Table Summary")
print("------------------------")
latex_table = r"""\begin{table}[t]
\centering
\small
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Bengali} & \textbf{English} \\
\midrule
Mean Toxicity   & %.2f & %.2f \\
Median Toxicity & %.1f & %.1f \\
\midrule
Score 1 (%%)    & %.1f & %.1f \\
Score 2 (%%)    & %.1f & %.1f \\
Score 3 (%%)    & %.1f & %.1f \\
Score 4 (%%)    & %.1f & %.1f \\
Score 5 (%%)    & %.1f & %.1f \\
\midrule
Mann-Whitney $U$ & \multicolumn{2}{c}{%.3f} \\
$p$-value        & \multicolumn{2}{c}{%.4f} \\
\bottomrule
\end{tabular}
\caption{Comparison of toxicity ratings for 100 English and Bengali sentences annotated independently.}
\label{tab:toxicity_comparison}
\end{table}
""" % (
    bengali_mean, english_mean,
    bengali_median, english_median,
    bengali_dist[1], english_dist[1],
    bengali_dist[2], english_dist[2],
    bengali_dist[3], english_dist[3],
    bengali_dist[4], english_dist[4],
    bengali_dist[5], english_dist[5],
    stat, pval
)

print(latex_table)
