"""
Applied Data Analysis: WHO Life Expectancy (2000-2015)

Question: Which health and socioeconomic factors are most strongly associated
with life expectancy across countries, and does the association with schooling
differ between developed and developing countries?

Run:  python analysis.py
Data: put "Life Expectancy Data.csv" (Kaggle: kumarajarshi/life-expectancy-who)
      in ./data/
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/Life Expectancy Data.csv")
OUT = Path("outputs")
(OUT / "figures").mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------------
# 1. Load and clean
# ----------------------------------------------------------------------------
df = pd.read_csv(DATA)
# Column names have stray spaces / mixed case -> snake_case
df.columns = [re.sub(r"[^0-9a-z]+", "_", c.strip().lower()).strip("_") for c in df.columns]
print("Raw shape:", df.shape)
print("Missing values (%):\n", (df.isna().mean() * 100).round(1).sort_values(ascending=False).head(8))

df = df.dropna(subset=["life_expectancy"])

# Zeros in these columns are placeholders for "missing", not real values
for col in ["income_composition_of_resources", "schooling"]:
    df.loc[df[col] == 0, col] = np.nan

# BMI is not plausible as a country-level average (values >60 and <10 are common)
# and the two child-mortality columns are near-duplicates -> drop for clarity
df = df.drop(columns=["bmi", "under_five_deaths", "population", "measles"])

# Impute within country by linear interpolation over time, then by status-year median
df = df.sort_values(["country", "year"])
num_cols = df.select_dtypes("number").columns.difference(["year", "life_expectancy"])
df[num_cols] = df.groupby("country")[num_cols].transform(lambda s: s.interpolate(limit_direction="both"))
df[num_cols] = df.groupby(["status", "year"])[num_cols].transform(lambda s: s.fillna(s.median()))
df = df.dropna()

df["log_gdp"] = np.log(df["gdp"].clip(lower=1))
df["log_hiv"] = np.log1p(df["hiv_aids"])
df["developed"] = (df["status"] == "Developed").astype(int)
print("Clean shape:", df.shape, "| countries:", df["country"].nunique())

# ----------------------------------------------------------------------------
# 2. Exploratory analysis
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["life_expectancy"], kde=True, ax=ax[0])
ax[0].set_title("Distribution of life expectancy")
sns.lineplot(data=df, x="year", y="life_expectancy", hue="status", errorbar="sd", ax=ax[1])
ax[1].set_title("Trend by development status (mean ± SD)")
plt.tight_layout(); plt.savefig(OUT / "figures/fig1_distribution_trend.png", dpi=200); plt.close()

feats = ["adult_mortality", "log_hiv", "schooling", "income_composition_of_resources",
         "log_gdp", "alcohol", "polio", "diphtheria", "hepatitis_b", "total_expenditure",
         "thinness_1_19_years"]
plt.figure(figsize=(8, 6.5))
sns.heatmap(df[["life_expectancy"] + feats].corr(), annot=True, fmt=".2f", cmap="vlag", center=0, annot_kws={"size": 7})
plt.title("Correlation matrix")
plt.tight_layout(); plt.savefig(OUT / "figures/fig2_correlations.png", dpi=200); plt.close()

plt.figure(figsize=(6.5, 4.5))
sns.scatterplot(data=df, x="schooling", y="life_expectancy", hue="status", alpha=0.5)
plt.title("Schooling vs life expectancy")
plt.tight_layout(); plt.savefig(OUT / "figures/fig3_schooling_scatter.png", dpi=200); plt.close()

# ----------------------------------------------------------------------------
# 3a. Inference: mixed-effects model (random intercept + random year slope per country)
#     Repeated yearly observations per country are not independent, so plain
#     OLS would understate standard errors; random slopes let each country
#     have its own time trend.
# ----------------------------------------------------------------------------
# adult_mortality is excluded: it is nearly the inverse of life expectancy by construction.
# year is included to absorb the general upward trend over time.
model_feats = ["year", "log_hiv", "schooling", "log_gdp", "polio", "diphtheria",
               "total_expenditure", "alcohol"]
z = df.copy()
z[model_feats] = (z[model_feats] - z[model_feats].mean()) / z[model_feats].std()  # standardize

formula = "life_expectancy ~ " + " + ".join(model_feats) + " + developed + schooling:developed"
mixed = smf.mixedlm(formula, z, groups=z["country"], re_formula="~year").fit(reml=True)
print(mixed.summary())

coef = pd.DataFrame({"coef": mixed.fe_params, "se": mixed.bse_fe, "p": mixed.pvalues[mixed.fe_params.index]})
coef["ci_low"] = coef["coef"] - 1.96 * coef["se"]
coef["ci_high"] = coef["coef"] + 1.96 * coef["se"]
coef.round(3).to_csv(OUT / "mixed_model_coefficients.csv")

c = coef.drop("Intercept").sort_values("coef")
plt.figure(figsize=(7, 5))
plt.errorbar(c["coef"], range(len(c)), xerr=1.96 * c["se"], fmt="o", capsize=3)
plt.yticks(range(len(c)), c.index); plt.axvline(0, color="grey", ls="--")
plt.xlabel("Effect on life expectancy (years per 1 SD)")
plt.title("Mixed-effects model: fixed effects (95% CI)")
plt.tight_layout(); plt.savefig(OUT / "figures/fig4_coefficients.png", dpi=200); plt.close()

# Residual diagnostics
resid, fitted = mixed.resid, mixed.fittedvalues
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(fitted, resid, alpha=0.3, s=10)
ax[0].axhline(0, color="grey", ls="--")
ax[0].set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")
sm.qqplot(resid, line="s", ax=ax[1], markersize=3)
ax[1].set_title("Q-Q plot of residuals")
plt.tight_layout(); plt.savefig(OUT / "figures/fig6_residual_diagnostics.png", dpi=200); plt.close()

# Largest residuals and sensitivity check (refit without them)
top = resid.abs().nlargest(10).index
print(df.loc[top, ["country", "year", "life_expectancy"]])
z2 = z.drop(top)
m2 = smf.mixedlm(formula, z2, groups=z2["country"], re_formula="~year").fit(reml=True)
sens = pd.DataFrame({
    "main_coef": mixed.fe_params, "main_p": mixed.pvalues[mixed.fe_params.index],
    "sens_coef": m2.fe_params, "sens_p": m2.pvalues[m2.fe_params.index],
})
print(sens.round(3))
sens.round(3).to_csv(OUT / "sensitivity_comparison.csv")
print("Countries with residuals above +5:")

# Which countries drive the positive-residual streaks in Fig. 6?
print(df.assign(resid=resid).query("resid > 5").groupby("country").size().sort_values(ascending=False).head(10))

# ----------------------------------------------------------------------------
# 3b. Validity check: predictive performance with country-grouped CV
#     (random splits would leak the same country into train and test)
# ----------------------------------------------------------------------------
X, y, groups = df[model_feats + ["developed"]], df["life_expectancy"], df["country"]
models = {
    "Lasso": make_pipeline(StandardScaler(), LassoCV(cv=5)),
    "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1),
}
rows = []
for name, m in models.items():
    r2s, rmses = [], []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        r2s.append(r2_score(y.iloc[te], pred))
        rmses.append(mean_squared_error(y.iloc[te], pred) ** 0.5)
    rows.append({"model": name, "R2": np.mean(r2s), "RMSE": np.mean(rmses)})
cv = pd.DataFrame(rows).round(3)
print(cv)
cv.to_csv(OUT / "cv_results.csv", index=False)

rf = models["Random forest"].fit(X, y)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(6.5, 4.5)); imp.plot.barh()
plt.title("Random forest feature importance"); plt.xlabel("Importance")
plt.tight_layout(); plt.savefig(OUT / "figures/fig5_rf_importance.png", dpi=200); plt.close()

print("Done. Results in ./outputs")