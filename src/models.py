from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import f1_score
import numpy as np


@dataclass
class Preprocessor:
    numeric_cols: List[str]
    categorical_cols: List[str]

    def build(self) -> ColumnTransformer:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        # scikit-learn >=1.4 usa 'sparse_output'; mantener compatibilidad
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ])
        pre = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.numeric_cols),
                ("cat", cat_pipe, self.categorical_cols),
            ]
        )
        return pre


def build_classifiers(pre: ColumnTransformer) -> Dict[str, Pipeline]:
    models = {
        'LR': LogisticRegression(max_iter=2000, n_jobs=None),
        'RF': RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
        'HGB': HistGradientBoostingClassifier(random_state=42),
    }
    out = {name: Pipeline(steps=[("pre", pre), ("model", mdl)]) for name, mdl in models.items()}
    return out


def build_regressors(pre: ColumnTransformer) -> Dict[str, Pipeline]:
    models = {
        'ridge': Ridge(alpha=1.0),
        'rf': RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
        'hgbr': HistGradientBoostingRegressor(random_state=42),
    }
    out = {name: Pipeline(steps=[("pre", pre), ("model", mdl)]) for name, mdl in models.items()}
    return out


def time_based_split(df: pd.DataFrame, val_month: str) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve índices (train_idx, val_idx) para un mes de validación fijo."""
    vm = pd.Period(val_month, freq='M')
    val_mask = df['target_month'] == vm
    train_mask = df['target_month'] < vm
    return df.index[train_mask].to_numpy(), df.index[val_mask].to_numpy()


def per_cut_splits(df: pd.DataFrame, val_months: List[str]) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    splits = []
    for m in val_months:
        tr, va = time_based_split(df, m)
        splits.append((tr, va, m))
    return splits


def evaluate_classifiers(df: pd.DataFrame, feature_cols: List[str], pre: ColumnTransformer,
                         val_months: List[str]) -> pd.DataFrame:
    X = df[feature_cols]
    y = df['y_cls'].astype(int)
    models = build_classifiers(pre)
    records = []
    for tr_idx, va_idx, month in per_cut_splits(df, val_months):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        row = {'cut_month': month}
        for name, pipe in models.items():
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xva)
            f1 = f1_score(yva, pred)
            row[name] = f1
        records.append(row)
    return pd.DataFrame.from_records(records)


def evaluate_regressors(df: pd.DataFrame, feature_cols: List[str], pre: ColumnTransformer,
                        val_month: str) -> pd.DataFrame:
    X = df[feature_cols]
    y = df['y_reg'].astype(float)
    tr_idx, va_idx = time_based_split(df, val_month)
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    models = build_regressors(pre)
    row = {'val_month': val_month}
    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xva)
        rmse = float(np.sqrt(np.mean((yva - pred) ** 2)))
        row[name] = rmse
    return pd.DataFrame([row])


def fit_best_regressor(df: pd.DataFrame, feature_cols: List[str], pre: ColumnTransformer,
                       fixed_val_month: str = '2023-06') -> Tuple[str, Pipeline]:
    # Selección por RMSE en validación fija
    X = df[feature_cols]
    y = df['y_reg'].astype(float)
    tr_idx, va_idx = time_based_split(df, fixed_val_month)
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    models = build_regressors(pre)
    best_name, best_pipe, best_rmse = None, None, float('inf')
    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xva)
        rmse = float(np.sqrt(np.mean((yva - pred) ** 2)))
        if rmse < best_rmse:
            best_name, best_pipe, best_rmse = name, pipe, rmse
    # Reentrena con todo el set hasta fixed_val_month (excluyendo ese mes)
    all_train_idx = df.index[df['target_month'] < pd.Period(fixed_val_month, freq='M')].to_numpy()
    best_pipe.fit(X.iloc[all_train_idx], y.iloc[all_train_idx])
    return best_name, best_pipe


def fit_best_classifier(df: pd.DataFrame, feature_cols: List[str], pre: ColumnTransformer,
                        fixed_val_month: str = '2023-06') -> Tuple[str, Pipeline]:
    # Selección por F1 en validación fija
    X = df[feature_cols]
    y = df['y_cls'].astype(int)
    tr_idx, va_idx = time_based_split(df, fixed_val_month)
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    models = build_classifiers(pre)
    best_name, best_pipe, best_f1 = None, None, -1.0
    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xva)
        f1 = f1_score(yva, pred)
        if f1 > best_f1:
            best_name, best_pipe, best_f1 = name, pipe, f1
    # Reentrena con todo el set hasta fixed_val_month (excluyendo ese mes)
    all_train_idx = df.index[df['target_month'] < pd.Period(fixed_val_month, freq='M')].to_numpy()
    best_pipe.fit(X.iloc[all_train_idx], y.iloc[all_train_idx])
    return best_name, best_pipe


def fit_all_classifiers_full(df: pd.DataFrame, feature_cols: List[str], pre: ColumnTransformer,
                             train_end: str = '2023-06') -> Dict[str, Pipeline]:
    X = df[df['target_month'] <= pd.Period(train_end, freq='M')][feature_cols]
    y = df[df['target_month'] <= pd.Period(train_end, freq='M')]['y_cls'].astype(int)
    models = build_classifiers(pre)
    for name, pipe in models.items():
        pipe.fit(X, y)
    return models


def fit_all_regressors_full(df: pd.DataFrame, feature_cols: List[str], pre: ColumnTransformer,
                            train_end: str = '2023-06') -> Dict[str, Pipeline]:
    X = df[df['target_month'] <= pd.Period(train_end, freq='M')][feature_cols]
    y = df[df['target_month'] <= pd.Period(train_end, freq='M')]['y_reg'].astype(float)
    models = build_regressors(pre)
    for name, pipe in models.items():
        pipe.fit(X, y)
    return models
