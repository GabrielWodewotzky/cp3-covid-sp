"""Módulo preditivo do dashboard: previsão semanal de casos/óbitos novos de COVID-19.

Abordagem: regressão supervisionada sobre série temporal, com atributos defasados
(lags de 1 a 4 semanas, média das últimas 4 semanas e tendência recente).
A avaliação usa split temporal (as últimas N semanas ficam de teste, sem embaralhar)
e é comparada com um baseline ingênuo ("a próxima semana será igual à última").
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LAGS = (1, 2, 3, 4)
COLUNAS = [f"lag_{l}" for l in LAGS] + ["media_4s", "tendencia"]

MODELOS = {
    "Regressão Ridge": lambda: Ridge(alpha=1.0),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=42),
}


def serie_semanal(diario: pd.Series) -> pd.Series:
    """Agrega uma série diária (índice datetime) em semanas terminando no sábado."""
    return diario.clip(lower=0).resample("W-SAT").sum()


def montar_atributos(s: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame({f"lag_{l}": s.shift(l) for l in LAGS})
    X["media_4s"] = s.shift(1).rolling(4).mean()
    X["tendencia"] = s.shift(1) - s.shift(2)
    X["y"] = s
    return X.dropna()


def metricas(y_real, y_prev) -> dict:
    y_real, y_prev = np.asarray(y_real, float), np.asarray(y_prev, float)
    wape = np.abs(y_real - y_prev).sum() / max(np.abs(y_real).sum(), 1e-9) * 100
    return {
        "MAE": mean_absolute_error(y_real, y_prev),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_prev))),
        "R²": r2_score(y_real, y_prev) if len(y_real) > 1 else float("nan"),
        "WAPE (%)": wape,
    }


def avaliar(s: pd.Series, nome_modelo: str, n_teste: int = 12):
    """Treina no passado e testa nas últimas n_teste semanas. Retorna (df_teste, métricas, métricas_baseline, modelo)."""
    d = montar_atributos(s)
    if len(d) < n_teste + 20:
        raise ValueError("Série curta demais para treinar e testar com segurança.")
    treino, teste = d.iloc[:-n_teste], d.iloc[-n_teste:]
    modelo = MODELOS[nome_modelo]()
    modelo.fit(treino[COLUNAS], treino["y"])
    prev = np.clip(modelo.predict(teste[COLUNAS]), 0, None)
    resultado = pd.DataFrame({"real": teste["y"], "previsto": prev, "baseline": teste["lag_1"]})
    return (
        resultado,
        metricas(resultado["real"], resultado["previsto"]),
        metricas(resultado["real"], resultado["baseline"]),
        modelo,
    )


def prever_futuro(s: pd.Series, nome_modelo: str, horizonte: int = 4) -> pd.Series:
    """Reajusta o modelo com toda a série e prevê as próximas semanas de forma recursiva."""
    d = montar_atributos(s)
    modelo = MODELOS[nome_modelo]()
    modelo.fit(d[COLUNAS], d["y"])
    hist = s.copy().astype(float)
    for _ in range(horizonte):
        v = hist.values
        linha = {f"lag_{l}": v[-l] for l in LAGS}
        linha["media_4s"] = v[-4:].mean()
        linha["tendencia"] = v[-1] - v[-2]
        p = max(float(modelo.predict(pd.DataFrame([linha])[COLUNAS])[0]), 0.0)
        hist.loc[hist.index[-1] + pd.Timedelta(weeks=1)] = p
    return hist.iloc[-horizonte:]


def importancias(modelo) -> pd.Series | None:
    if hasattr(modelo, "feature_importances_"):
        return pd.Series(modelo.feature_importances_, index=COLUNAS).sort_values()
    if hasattr(modelo, "coef_"):
        return pd.Series(np.abs(modelo.coef_), index=COLUNAS).sort_values()
    return None
