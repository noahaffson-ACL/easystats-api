"""Statistical engine for ThèsIA: descriptive tables, bivariate tests and
multivariable regression, built on pandas / scipy / statsmodels."""

from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _classify_variable(series: pd.Series) -> str:
    """Infer a variable's role: 'categorielle', 'quantitative_continue' or 'texte'."""
    s = series.dropna()
    unique_vals = s.nunique()
    if _is_numeric(s):
        return "categorielle" if unique_vals <= 5 else "quantitative_continue"
    return "categorielle" if unique_vals <= 10 else "texte"


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
        var_type = _classify_variable(df[col])

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


# ── Effect sizes ───────────────────────────────────────

def _cohens_d(s1: np.ndarray, s2: np.ndarray) -> float:
    n1, n2 = len(s1), len(s2)
    pooled_var = ((n1 - 1) * np.var(s1, ddof=1) + (n2 - 1) * np.var(s2, ddof=1)) / (n1 + n2 - 2)
    if pooled_var == 0:
        return 0.0
    return float((np.mean(s1) - np.mean(s2)) / np.sqrt(pooled_var))


def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    return float(1 - (2 * u_stat) / (n1 * n2))


def _eta_squared_anova(samples: List[np.ndarray]) -> float:
    all_vals = np.concatenate(samples)
    grand_mean = all_vals.mean()
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    if ss_total == 0:
        return 0.0
    ss_between = sum(len(s) * (np.mean(s) - grand_mean) ** 2 for s in samples)
    return float(ss_between / ss_total)


def _eta_squared_kruskal(h_stat: float, n: int, k: int) -> float:
    if n - k == 0:
        return 0.0
    return float((h_stat - k + 1) / (n - k))


def _cramers_v(chi2: float, contingency: pd.DataFrame) -> float:
    n = contingency.values.sum()
    r, c = contingency.shape
    denom = n * (min(r, c) - 1)
    if denom == 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))


def _odds_ratio_2x2(contingency: pd.DataFrame):
    table = contingency.to_numpy(dtype=float)
    a, b = table[0, 0], table[0, 1]
    c, d = table[1, 0], table[1, 1]
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_val = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ic95 = [float(or_val * np.exp(-1.96 * se)), float(or_val * np.exp(1.96 * se))]
    return float(or_val), ic95


def _interpret_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "négligeable"
    if d < 0.5:
        return "faible"
    if d < 0.8:
        return "moyen"
    return "fort"


def _interpret_r(r: float) -> str:
    r = abs(r)
    if r < 0.1:
        return "négligeable"
    if r < 0.3:
        return "faible"
    if r < 0.5:
        return "modérée"
    if r < 0.7:
        return "forte"
    return "très forte"


def _interpret_eta2(eta2: float) -> str:
    eta2 = abs(eta2)
    if eta2 < 0.01:
        return "négligeable"
    if eta2 < 0.06:
        return "faible"
    if eta2 < 0.14:
        return "moyen"
    return "fort"


def _interpret_cramers_v(v: float) -> str:
    if v < 0.1:
        return "négligeable"
    if v < 0.3:
        return "faible"
    if v < 0.5:
        return "moyen"
    return "fort"


# ── Plain-language interpretation ──────────────────────

def _phrase_significativite(p_val: float, significatif: bool, sujet: str = "Cette différence") -> str:
    if significatif:
        return (
            f"{sujet} est statistiquement significative (p = {p_val}) : "
            "elle est probablement réelle, et pas due au hasard."
        )
    return (
        f"{sujet} n'est pas statistiquement significative (p = {p_val}) : "
        "elle pourrait simplement être due au hasard, on ne peut pas conclure."
    )


def _phrase_correlation(var_dep: str, var: str, r: float, p_val: float,
                         significatif: bool, force: str) -> str:
    direction = "augmenter" if r > 0 else "diminuer"
    phrase = f"Plus « {var} » augmente, plus « {var_dep} » a tendance à {direction} (lien {force}). "
    return phrase + _phrase_significativite(p_val, significatif, sujet="Cette relation")


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
        effect_size = None

        if n_groupes == 2:
            if normal:
                stat, p_val = stats.ttest_ind(*samples)
                test_nom = "t-test de Student"
                d = _cohens_d(samples[0], samples[1])
                effect_size = {"nom": "d de Cohen", "valeur": round(d, 4), "interpretation": _interpret_d(d)}
            else:
                stat, p_val = stats.mannwhitneyu(*samples)
                test_nom = "Mann-Whitney U"
                r = _rank_biserial(float(stat), len(samples[0]), len(samples[1]))
                effect_size = {"nom": "corrélation rang-bisériale", "valeur": round(r, 4), "interpretation": _interpret_r(r)}
        else:
            if normal:
                stat, p_val = stats.f_oneway(*samples)
                test_nom = "ANOVA à un facteur"
                eta2 = _eta_squared_anova(samples)
                effect_size = {"nom": "eta carré", "valeur": round(eta2, 4), "interpretation": _interpret_eta2(eta2)}
            else:
                stat, p_val = stats.kruskal(*samples)
                test_nom = "Test de Kruskal-Wallis"
                eta2 = _eta_squared_kruskal(float(stat), len(sub), n_groupes)
                effect_size = {"nom": "eta carré (approx.)", "valeur": round(eta2, 4), "interpretation": _interpret_eta2(eta2)}

        result = {
            "test": test_nom,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 4),
            "significatif": bool(p_val < 0.05),
            "groupes": [str(g) for g in groupes]
        }
        if effect_size is not None:
            result["effect_size"] = effect_size

        means = {str(g): float(np.mean(s)) for g, s in zip(groupes, samples)}
        sorted_means = sorted(means.items(), key=lambda x: x[1], reverse=True)
        if n_groupes == 2:
            diff = sorted_means[0][1] - sorted_means[1][1]
            interpretation = (
                f"En moyenne, « {var_dep} » est plus élevé chez « {sorted_means[0][0]} » "
                f"({sorted_means[0][1]:.2f}) que chez « {sorted_means[1][0]} » "
                f"({sorted_means[1][1]:.2f}) — différence de {diff:.2f}. "
            )
        else:
            interpretation = (
                f"« {var_dep} » varie selon les groupes de « {var_groupe} » : moyenne la plus "
                f"élevée chez « {sorted_means[0][0]} » ({sorted_means[0][1]:.2f}), la plus basse "
                f"chez « {sorted_means[-1][0]} » ({sorted_means[-1][1]:.2f}). "
            )
        result["interpretation"] = interpretation + _phrase_significativite(result["p_value"], result["significatif"])
        return result

    # Categorical outcome
    contingency = pd.crosstab(sub[var_groupe], sub[var_dep])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    stat = chi2
    test_nom = "Chi² de Pearson"
    effect_size = None

    if (expected < 5).any():
        if contingency.shape == (2, 2):
            _, p_val = stats.fisher_exact(contingency)
            test_nom = "Test exact de Fisher"
            stat = None
        else:
            test_nom = "Chi² de Pearson (effectifs attendus < 5 dans certaines cellules, à interpréter avec prudence)"

    if contingency.shape == (2, 2):
        or_val, ic95 = _odds_ratio_2x2(contingency)
        effect_size = {"nom": "odds ratio", "valeur": round(or_val, 4), "ic95": [round(b, 4) for b in ic95]}
    else:
        v = _cramers_v(chi2, contingency)
        effect_size = {"nom": "V de Cramér", "valeur": round(v, 4), "interpretation": _interpret_cramers_v(v)}

    result = {
        "test": test_nom,
        "statistic": round(float(stat), 4) if stat is not None else None,
        "p_value": round(float(p_val), 4),
        "significatif": bool(p_val < 0.05),
        "groupes": [str(g) for g in groupes],
        "effect_size": effect_size
    }

    levels_dep = sorted(sub[var_dep].unique(), key=str)
    if len(levels_dep) == 2:
        event = levels_dep[1]
        props = {}
        for g in groupes:
            gs = sub[sub[var_groupe] == g]
            props[str(g)] = float((gs[var_dep] == event).mean() * 100)
        sorted_props = sorted(props.items(), key=lambda x: x[1], reverse=True)
        interpretation = (
            f"« {event} » est observé chez {sorted_props[0][1]:.1f}% du groupe « {sorted_props[0][0]} » "
            f"contre {sorted_props[-1][1]:.1f}% du groupe « {sorted_props[-1][0]} ». "
        )
    else:
        interpretation = f"La répartition de « {var_dep} » diffère selon les groupes de « {var_groupe} ». "

    result["interpretation"] = interpretation + _phrase_significativite(result["p_value"], result["significatif"])
    return result


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
        coef_val = round(float(transform(model.params[var])), 4)
        p_value = round(float(model.pvalues[var]), 4)
        significatif = p_value < 0.05

        if var == "const":
            interpretation = "Valeur de référence du modèle (constante), non interprétable isolément."
        elif model_type == "régression linéaire":
            direction = "augmentation" if coef_val >= 0 else "diminution"
            interpretation = (
                f"Toutes choses égales par ailleurs, chaque unité supplémentaire de « {var} » est "
                f"associée à une {direction} de {abs(coef_val):.2f} de « {var_dep} ». "
            ) + _phrase_significativite(p_value, significatif, sujet="Cette association")
        else:
            if coef_val > 1:
                interpretation = (
                    f"Toutes choses égales par ailleurs, « {var} » multiplie par {coef_val:.2f} "
                    f"les chances de « {levels[1]} » (par rapport à « {levels[0]} »). "
                )
            elif coef_val < 1:
                interpretation = (
                    f"Toutes choses égales par ailleurs, « {var} » divise par {1 / coef_val:.2f} "
                    f"les chances de « {levels[1]} » (effet protecteur par rapport à « {levels[0]} »). "
                )
            else:
                interpretation = f"« {var} » n'a pas d'effet notable sur les chances de « {levels[1]} ». "
            interpretation += _phrase_significativite(p_value, significatif, sujet="Cette association")

        coefficients.append({
            "variable": var,
            coef_label: coef_val,
            "ic95": [
                round(float(transform(conf_int.loc[var, 0])), 4),
                round(float(transform(conf_int.loc[var, 1])), 4)
            ],
            "p_value": p_value,
            "interpretation": interpretation
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


def correlation_analysis(df: pd.DataFrame, var_dep: str, vars_indep: List[str]) -> List[dict]:
    """Pairwise correlation (Pearson or Spearman, depending on normality)
    between `var_dep` and each numeric variable in `vars_indep`."""
    if not _is_numeric(df[var_dep]):
        raise ValueError("La variable dépendante doit être quantitative pour une analyse de corrélation")

    correlations = []
    for var in vars_indep:
        if var not in df.columns:
            raise ValueError(f"Variable inconnue : {var}")

        if not _is_numeric(df[var]):
            correlations.append({"variable": var, "erreur": "variable non quantitative, corrélation ignorée"})
            continue

        sub = df[[var_dep, var]].dropna()
        if len(sub) < 3:
            correlations.append({"variable": var, "erreur": "pas assez de données"})
            continue

        normal = _normal_groups([sub[var_dep].to_numpy(), sub[var].to_numpy()])
        if normal:
            r, p_val = stats.pearsonr(sub[var_dep], sub[var])
            methode = "Pearson"
        else:
            r, p_val = stats.spearmanr(sub[var_dep], sub[var])
            methode = "Spearman"

        r_rounded = round(float(r), 4)
        p_rounded = round(float(p_val), 4)
        significatif = bool(p_val < 0.05)
        force = _interpret_r(float(r))

        correlations.append({
            "variable": var,
            "methode": methode,
            "r": r_rounded,
            "p_value": p_rounded,
            "significatif": significatif,
            "n": int(len(sub)),
            "force": force,
            "interpretation": _phrase_correlation(var_dep, var, r_rounded, p_rounded, significatif, force)
        })

    return correlations


def run_analysis(data: List[dict], var_dep: str, vars_indep: List[str],
                  type_etude: Optional[str] = None,
                  groupes: Optional[str] = None) -> dict:
    """Run the full analysis pipeline shared by /analyze and /export-docx.

    `type_etude` selects the analysis run on `vars_indep`:
    - "correlation": pairwise Pearson/Spearman correlations
    - anything else (default): multivariable regression
    """
    df = pd.DataFrame(data)
    if var_dep not in df.columns:
        raise ValueError(f"Variable dépendante inconnue : {var_dep}")

    results: dict = {"tableau_descriptif": descriptive_table(df)}

    if groupes and groupes in df.columns:
        test = bivariate_test(df, var_dep, groupes)
        if test is not None:
            results["test_principal"] = test

    if vars_indep:
        if type_etude == "correlation":
            try:
                results["correlations"] = correlation_analysis(df, var_dep, vars_indep)
            except ValueError as exc:
                results["correlations_erreur"] = str(exc)
        else:
            try:
                results["regression"] = regression_analysis(df, var_dep, vars_indep)
            except ValueError as exc:
                results["regression_erreur"] = str(exc)

    return results


def run_auto_analysis(data: List[dict], variable_principale: str) -> dict:
    """Fully automatic analysis: given only the data and the variable the
    student wants to study, test its association with every other variable
    in the dataset (choosing the appropriate test automatically) and fit an
    adjusted multivariable model. Designed so a student needs no statistical
    knowledge to get a complete, usable set of results."""
    df = pd.DataFrame(data)
    if variable_principale not in df.columns:
        raise ValueError(f"Variable inconnue : {variable_principale}")

    type_principale = _classify_variable(df[variable_principale])
    if type_principale == "texte":
        raise ValueError(
            f"La variable '{variable_principale}' semble être un champ de texte libre "
            "et ne peut pas être analysée statistiquement. Choisissez une variable "
            "numérique ou une variable à choix (catégorielle)."
        )

    results: dict = {
        "variable_principale": variable_principale,
        "type_variable_principale": type_principale,
        "tableau_descriptif": descriptive_table(df),
    }

    autres = [c for c in df.columns if c != variable_principale and not c.startswith("_")]

    associations = []
    for var in autres:
        type_var = _classify_variable(df[var])
        if type_var == "texte":
            continue

        try:
            if type_var == "categorielle":
                test = bivariate_test(df, variable_principale, var)
                if test is None:
                    continue
                association = {"variable": var, **test}
            elif type_principale == "quantitative_continue":
                corr = correlation_analysis(df, variable_principale, [var])[0]
                if "erreur" in corr:
                    continue
                association = {
                    "variable": var,
                    "test": f"Corrélation de {corr['methode']}",
                    "statistic": corr["r"],
                    "p_value": corr["p_value"],
                    "significatif": corr["significatif"],
                    "effect_size": {"nom": "r", "valeur": corr["r"], "interpretation": corr["force"]},
                    "interpretation": corr["interpretation"]
                }
            else:
                test = bivariate_test(df, var, variable_principale)
                if test is None:
                    continue
                association = {"variable": var, **test}
        except ValueError:
            continue

        associations.append(association)

    n_tests = len(associations)
    for assoc in associations:
        p_corrigee = min(assoc["p_value"] * n_tests, 1.0) if n_tests else assoc["p_value"]
        assoc["p_value_corrigee"] = round(p_corrigee, 4)
        assoc["significatif_corrige"] = bool(p_corrigee < 0.05)

    associations.sort(key=lambda a: a["p_value"])
    results["associations"] = associations

    # Adjusted multivariable model using every other usable variable as predictor
    vars_regression = [v for v in autres if _classify_variable(df[v]) != "texte"]
    if vars_regression:
        try:
            results["regression"] = regression_analysis(df, variable_principale, vars_regression)
        except Exception as exc:
            results["regression_erreur"] = (
                "Le modèle ajusté n'a pas pu être calculé avec toutes les variables "
                f"disponibles ({exc}). Essayez avec moins de variables ou plus d'observations."
            )

    significatifs = [a for a in associations if a["significatif"]]
    significatifs_corriges = [a for a in associations if a["significatif_corrige"]]

    results["resume"] = {
        "n_variables_testees": n_tests,
        "n_associations_significatives": len(significatifs),
        "variables_significatives": [a["variable"] for a in significatifs],
        "phrases_significatives": [
            f"{a['variable']} : {a['interpretation']}" for a in significatifs
        ],
        "n_associations_significatives_apres_correction": len(significatifs_corriges),
        "variables_significatives_apres_correction": [a["variable"] for a in significatifs_corriges],
        "avertissement": (
            "Tester de nombreuses variables augmente le risque de faux positifs (comparaisons "
            "multiples). La colonne 'p_value_corrigee' applique une correction de Bonferroni : "
            "privilégiez-la pour vos conclusions principales, en particulier pour les variables "
            "qui n'étaient pas votre hypothèse de départ."
        ) if n_tests > 1 else None
    }

    return results
