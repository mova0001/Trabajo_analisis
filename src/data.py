import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict


def _parse_month_id(s: pd.Series) -> pd.Series:
    """Parsea month_id tipo 'YYYY_MM' o 'YYYY-MM' a pandas Period[M]."""
    s = s.astype(str).str.replace("_", "-", regex=False)
    return s.apply(lambda x: pd.Period(x, freq='M'))


def _format_month(p: pd.Period) -> str:
    return f"{p.year:04d}-{p.month:02d}"


def load_core_frames(base_dir: Path = Path('.')) -> Dict[str, pd.DataFrame]:
    """Carga los CSVs principales del proyecto."""
    def read(name, **kw):
        df = pd.read_csv(base_dir / name, **kw)
        return df

    stock = read('stock_data.csv')
    stock['stock_id'] = stock['stock_id'].astype(str)
    stock['month'] = _parse_month_id(stock['month_id'])

    index_df = read('monashIndex.csv')
    index_df['month'] = _parse_month_id(index_df['month_id'])

    company = read('company_info.csv')
    company['stock_id'] = company['stock_id'].astype(str)

    # Macros
    macros = {}
    macros['vix'] = read('vix_index.csv').rename(columns={'vix': 'vix'})
    macros['vix']['month'] = _parse_month_id(macros['vix']['month_id'])

    macros['us10y'] = read('us_10_year_treasury.csv').rename(columns={'10y_treasury': 'us10y'})
    macros['us10y']['month'] = _parse_month_id(macros['us10y']['month_id'])

    macros['us5y'] = read('us_5_year_treasury.csv').rename(columns={'5y_treasury': 'us5y'})
    macros['us5y']['month'] = _parse_month_id(macros['us5y']['month_id'])

    macros['infl'] = read('fed_inflation_rate.csv').rename(columns={'inflation_rate': 'inflation_rate'})
    macros['infl']['month'] = _parse_month_id(macros['infl']['month_id'])

    macros['fed_rate'] = read('fed_funds_rate.csv').rename(columns={'fed_rate': 'fed_rate'})
    macros['fed_rate']['month'] = _parse_month_id(macros['fed_rate']['month_id'])

    macros['unemp'] = read('fed_unemployment_rate.csv').rename(columns={'unemployment_rate': 'unemployment_rate'})
    macros['unemp']['month'] = _parse_month_id(macros['unemp']['month_id'])

    # Targets
    train_tgt = read('training_targets.csv')
    train_tgt['stock_id'] = train_tgt['stock_id'].astype(str)
    train_tgt['month'] = _parse_month_id(train_tgt['month_id'])

    test_tgt = read('testing_targets.csv')
    test_tgt['stock_id'] = test_tgt['stock_id'].astype(str)
    test_tgt['month'] = _parse_month_id(test_tgt['month_id'])

    return {
        'stock': stock,
        'index': index_df,
        'company': company,
        'macros': macros,
        'train_tgt': train_tgt,
        'test_tgt': test_tgt,
    }


def build_feature_table(frames: Dict[str, pd.DataFrame],
                        macro_lag_months: int = 1) -> pd.DataFrame:
    """Construye una tabla de features por (stock_id, target_month),
    usando datos de t-1 para predecir t. También añade macros (laggeadas) y rasgos estáticos.
    """
    stock = frames['stock'].copy()
    company = frames['company'].copy()
    macros = frames['macros']

    # Features base de acciones tomadas en month = t-1 y etiquetadas para predecir t
    stock = stock.copy()
    # Cálculo explícito de retorno del mes (para explorar) — no se usa como feature para t
    stock['stock_return_calc'] = (stock['month_end_close_usd'] - stock['month_start_open_usd']) / stock['month_start_open_usd']

    # Definir target_month = month + 1
    stock['target_month'] = stock['month'] + 1

    # Selección de columnas de stock como features (as-of t-1)
    stock_feat_cols = [
        'month_start_open_usd','month_end_close_usd','month_high_usd','month_low_usd',
        'monthly_volume','intramonth_return','return_1m','return_3m','return_6m',
        'intramonth_volatility','volatility_3m','volatility_6m','trading_days',
        'avg_volume_3m','volume_ratio','price_range_ratio','stock_return_calc'
    ]
    # Filtrar presencia de columnas por robustez
    stock_feat_cols = [c for c in stock_feat_cols if c in stock.columns]

    feats = stock[['stock_id','month','target_month'] + stock_feat_cols].copy()

    # Añadir macros de t-1 (lag configurable)
    def merge_macro(df: pd.DataFrame, mdf: pd.DataFrame, colname: str) -> pd.DataFrame:
        tmp = mdf[['month', colname]].copy()
        # Queremos macro en (t - macro_lag_months) para features t-1. Como df está en month=t-1 y target=t,
        # macro debería corresponder a df.month - (macro_lag_months - 1). Si macro_lag_months=1, macro es df.month.
        tmp = tmp.rename(columns={'month': 'macro_month'})
        joined = df.copy()
        joined['macro_month'] = joined['month'] - (macro_lag_months - 1)
        return joined.merge(tmp, how='left', on='macro_month').drop(columns=['macro_month'])

    macro_map = {
        'vix': macros['vix'],
        'us10y': macros['us10y'],
        'us5y': macros['us5y'],
        'inflation_rate': macros['infl'],
        'fed_rate': macros['fed_rate'],
        'unemployment_rate': macros['unemp'],
    }
    for col, mdf in macro_map.items():
        feats = merge_macro(feats, mdf, col)

    # Rasgos estáticos por empresa
    feats = feats.merge(company, on='stock_id', how='left')

    return feats


def assemble_train_test(frames: Dict[str, pd.DataFrame],
                        macro_lag_months: int = 1,
                        train_end: str = '2023-06',
                        predict_month: str = '2023-07') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Genera Xy de entrenamiento (hasta train_end) y X de predicción para predict_month.
    Retorna (train, X_pred, test_targets_df)
    train: DataFrame con columnas ['stock_id','target_month','y_cls','y_reg', <features...>]
    X_pred: DataFrame con filas de predict_month (sin y)
    test_targets_df: DataFrame de testing_targets para referencia de ids.
    """
    feats = build_feature_table(frames, macro_lag_months=macro_lag_months)
    train_tgt = frames['train_tgt'].copy()
    test_tgt = frames['test_tgt'].copy()

    # Alinear targets: y de mes t se une con features target_month=t
    train = feats.merge(
        train_tgt[['stock_id','month','outperform_binary','excess_return']],
        left_on=['stock_id','target_month'], right_on=['stock_id','month'], how='inner', suffixes=("","_y")
    ).drop(columns=['month_y'])
    train = train.rename(columns={'month': 'feature_month'})
    # Filtrar por rango máximo de entrenamiento
    train_end_period = pd.Period(train_end, freq='M')
    train = train[train['target_month'] <= train_end_period].copy()

    # Conjunto de predicción: predict_month
    pred_period = pd.Period(predict_month, freq='M')
    X_pred = feats[feats['target_month'] == pred_period].copy()
    # Garantiza que cubrimos los ids de test (616)
    X_pred = test_tgt[['stock_id','month']].merge(
        X_pred.drop(columns=['month']).rename(columns={'target_month':'month'}),
        on=['stock_id','month'], how='left'
    ).rename(columns={'month':'target_month'})

    # Renombra columnas de y
    train = train.rename(columns={'outperform_binary':'y_cls','excess_return':'y_reg'})
    return train, X_pred, test_tgt.copy()


def split_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Devuelve (numeric_cols, categorical_cols) excluyendo identificadores y fechas."""
    exclude = {'stock_id','feature_month','target_month','y_cls','y_reg','month_id'}
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude).tolist()
    # Categóricas: las no numéricas que no sean ids
    cat_cols = [c for c in df.columns if c not in num_cols and c not in exclude]
    return num_cols, cat_cols

