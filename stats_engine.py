"""Statistical engine for ThèsIA: descriptive tables, bivariate tests and
multivariable regression, built on pandas / scipy / statsmodels."""

from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def detect_variables(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"variables": [], "n_observations": 0, "qualite_globale": "indéterminée"}

    variables = []
    for col in df.columns:
        if col.startswith("_"):
            continue

        series = df[col].dropna()
        missing_pct = round((df[col].isna().sum() / len(df)) * 100, 1)
        unique_vals = series.nunique()

        if _is_numeric(series):
            var_type = "categorielle" if unique_vals <= 5 else "quantitative_continue"
        else:
            var_type = "categorielle" if unique_vals <= 10 else "texte"

        variables.append({
            "name": col,
            "type": var_type,
            "missing_pct": missing_pct,
            "unique_values": int(unique_vals),
            "exemple_valeurs": series.head(3).tolist()
        })

    return {
        "variables": variables,
        "n_observations": len(df),
        "qualite_globale": "bonne" if df.isna().mean().mean() < 0.1 else "acceptable"
    }


def descriptive_table(df: pd.DataFrame) -> dict:
    tableau = {}
    for col in df.columns:
        if col.startswith("_"):
            continue

        series = df[col].dropna()
        if len(series) == 0:
            continue

        if _is_numeric(series):
            tableau[col] = {
                "type": "continue",
                "n": int(series.count()),
                "moyenne": round(float(series.mean()), 2),
                "ecart_type": round(float(series.std()), 2),
                "mediane": round(float(series.median()), 2),
                "q1": round(float(series.quantile(0.25)), 2),
                "q3": round(float(series.quantile(0.75)), 2)
            }
        else:
            freq = series.value_counts()
            tableau[col] = {
                "type": "categorielle",
                "n": int(series.count()),
                "frequences": {
                    str(k): {"n": int(v), "pct": round(v / len(series) * 100, 1)}
                    for k, v in freq.items()
                }
            }

    return tableau


def _normal_groups(samples: List[np.ndarray]) -> bool:
    """All groups are considered normal if each has >=3 observations and
    Shapiro-Wilk does not reject normality (p > 0.05)."""
    for sample in samples:
        if len(sample) < 3:
            return False
        _, p_normal = stats.shapiro(sample)
        if p_normal <= 0.05:
            return False
    return True


def bivariate_test(df: pd.DataFrame, var_dep: str, var_groupe: str) -> Optional[dict]:
    """Compare `var_dep` across the groups defined by `var_groupe`.

    Supports 2 groups (t-test/Mann-Whitney, Chi²/Fisher) as well as 3+ groups
    (ANOVA/Kruskal-Wallis, Chi² on a R x C contingency table).
    """
    if var_dep not in df.columns:
        raise ValueError(f"Variable dépendante inconnue : {var_dep}")
    if var_groupe not in df.columns:
        raise ValueError(f"Variable de groupe inconnue : {var_groupe}")

    sub = df[[var_dep, var_groupe]].dropna()
    groupes = sorted(sub[var_groupe].unique(), key=str)
    n_groupes = len(groupes)

    if n_groupes < 2:
        return None

    if _is_numeric(sub[var_dep]):
        samples = [sub.loc[sub[var_groupe] == g, var_dep].to_numpy() for g in groupes]
        if any(len(s) == 0 for s in samples):
            return None

        normal = _normal_groups(samples)

        if n_groupes == 2:
            if normal:
                stat, p_val = stats.ttest_ind(*samples)
                test_nom = "t-test de Student"
            else:
                stat, p_val = stats.mannwhitneyu(*samples)
                test_nom = "Mann-Whitney U"
        else:
            if normal:
                stat, p_val = stats.f_oneway(*samples)
                test_nom = "ANOVA à un facteur"
            else:
                stat, p_val = stats.kruskal(*samples)
                test_nom = "Test de Kruskal-Wallis"

        return {
            "test": test_nom,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 4),
            "significatif": bool(p_val < 0.05),
            "groupes": [str(g) for g in groupes]
        }

    # Categorical outcome
    contingency = pd.crosstab(sub[var_groupe], sub[var_dep])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    stat = chi2
    test_nom = "Chi² de Pearson"

    if (expected < 5).any():
        if contingency.shape == (2, 2):
            _, p_val = stats.fisher_exact(contingency)
            test_nom = "Test exact de Fisher"
            stat = None
        else:
            test_nom = "Chi² de Pearson (effectifs attendus < 5 dans certaines cellules, à interpréter avec prudence)"

    return {
        "test": test_nom,
        "statistic": round(float(stat), 4) if stat is not None else None,
        "p_value": round(float(p_val), 4),
        "significatif": bool(p_val < 0.05),
        "groupes": [str(g) for g in groupes]
    }


def regression_analysis(df: pd.DataFrame, var_dep: str, vars_indep: List[str]) -> dict:
    """Multivariable linear (OLS) or logistic regression of `var_dep` on
    `vars_indep`. The model is chosen automatically from the type of
    `var_dep`: logistic if binary, linear if continuous."""
    missing_cols = [c for c in [var_dep] + vars_indep if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Variable(s) inconnue(s) : {', '.join(missing_cols)}")

    sub = df[[var_dep] + vars_indep].dropna()
    if len(sub) < len(vars_indep) + 2:
        raise ValueError("Pas assez d'observations complètes pour estimer ce modèle")

    y = sub[var_dep]
    X = sub[vars_indep]

    # Encode categorical predictors as dummy variables
    X = pd.get_dummies(X, drop_first=True)
    if X.shape[1] == 0:
        raise ValueError("Aucune variable indépendante exploitable après encodage")
    X = sm.add_constant(X.astype(float))

    if _is_numeric(y) and y.nunique() > 5:
        model = sm.OLS(y.astype(float), X).fit()
        model_type = "régression linéaire"
        transform = lambda v: v
        coef_label = "coefficient"
    else:
        levels = sorted(y.unique(), key=str)
        if len(levels) != 2:
            raise ValueError(
                "La régression logistique nécessite une variable dépendante binaire (2 catégories)"
            )
        y_bin = (y == levels[1]).astype(float)
        model = sm.Logit(y_bin, X).fit(disp=0)
        model_type = "régression logistique"
        transform = np.exp
        coef_label = "odds_ratio"

    conf_int = model.conf_int()
    coefficients = []
    for var in model.params.index:
        coefficients.append({
            "variable": var,
            coef_label: round(float(transform(model.params[var])), 4),
            "ic95": [
                round(float(transform(conf_int.loc[var, 0])), 4),
                round(float(transform(conf_int.loc[var, 1])), 4)
            ],
            "p_value": round(float(model.pvalues[var]), 4),
        })

    result = {
        "type": model_type,
        "n": int(len(sub)),
        "coefficients": coefficients,
    }

    if model_type == "régression linéaire":
        result["r2"] = round(float(model.rsquared), 4)
        result["r2_ajuste"] = round(float(model.rsquared_adj), 4)
    else:
        result["pseudo_r2"] = round(float(model.prsquared), 4)
        result["categorie_reference"] = str(levels[0])
        result["categorie_evenement"] = str(levels[1])

    return result


def run_analysis(data: List[dict], var_dep: str, vars_indep: List[str],
                  groupes: Optional[str] = None) -> dict:
    """Run the full analysis pipeline shared by /analyze and /export-docx."""
    df = pd.DataFrame(data)
    if var_dep not in df.columns:
        raise ValueError(f"Variable dépendante inconnue : {var_dep}")

    results: dict = {"tableau_descriptif": descriptive_table(df)}

    if groupes and groupes in df.columns:
        test = bivariate_test(df, var_dep, groupes)
        if test is not None:
            results["test_principal"] = test

    if vars_indep:
        try:
            results["regression"] = regression_analysis(df, var_dep, vars_indep)
        except ValueError as exc:
            results["regression_erreur"] = str(exc)

    return results
