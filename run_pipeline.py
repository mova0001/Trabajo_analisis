from pathlib import Path
import argparse
import pandas as pd

from src.data import load_core_frames, assemble_train_test, split_feature_columns, _format_month
from src.models import (
    Preprocessor, evaluate_classifiers, evaluate_regressors,
    fit_best_classifier, fit_all_classifiers_full, fit_all_regressors_full,
    fit_best_regressor
)

def main():
    parser = argparse.ArgumentParser(description='Run pipeline for Monash relative performance task')
    parser.add_argument('--full', action='store_true', help='Generar métricas y archivos auxiliares (CV, preds intermedias)')
    parser.add_argument('--train-end', default='2023-06', help='Último mes de entrenamiento (YYYY-MM)')
    parser.add_argument('--predict-month', default='2023-07', help='Mes a predecir (YYYY-MM)')
    args = parser.parse_args()
    full_mode = args.full

    base = Path('.')
    frames = load_core_frames(base)

    # Construir X/y evitando leakage: features de t-1 para predecir t
    train_df, X_pred, test_tgt = assemble_train_test(frames, macro_lag_months=1,
                                                     train_end=args.train_end, predict_month=args.predict_month)

    num_cols, cat_cols = split_feature_columns(train_df)
    pre = Preprocessor(num_cols, cat_cols).build()
    feature_cols = num_cols + cat_cols

    # 1) Evaluación Clasificación (per-cuts y promedio) — solo en modo full
    if full_mode:
        per_cut = evaluate_classifiers(train_df, feature_cols, pre, val_months=['2022-06','2022-12', args.train_end])
        per_cut.to_csv('cv_per_cut_f1.csv', index=False)

        avg = pd.DataFrame({
            'Average_F1': per_cut[['LR','RF','HGB']].mean().values
        }, index=['LR','RF','HGB']).T
        avg.to_csv('cv_average_f1.csv')

        fixed = per_cut[per_cut['cut_month']==args.train_end][['LR','RF','HGB']]
        fixed.index = [f'F1@{args.train_end}']
        fixed.to_csv(f'cv_fixed_f1_{args.train_end}.csv')

        # 2) Evaluación Regresión en train_end (opcional)
        _ = evaluate_regressors(train_df, feature_cols, pre, val_month=args.train_end)

    # 3) Ajuste final y predicciones 2023-07
    best_name, best_clf = fit_best_classifier(train_df, feature_cols, pre, fixed_val_month=args.train_end)
    best_reg_name, best_reg = fit_best_regressor(train_df, feature_cols, pre, fixed_val_month=args.train_end)
    all_clfs = fit_all_classifiers_full(train_df, feature_cols, pre, train_end=args.train_end)
    all_regs = fit_all_regressors_full(train_df, feature_cols, pre, train_end=args.train_end)

    # Predicciones clasificación por modelo
    cls_pred = []
    for name, pipe in all_clfs.items():
        proba = pipe.predict_proba(X_pred[feature_cols])[:, 1]
        pred = (proba >= 0.5).astype(int)
        tmp = X_pred[['stock_id','target_month']].copy()
        tmp[f'proba_{name.lower()}'] = proba
        tmp[f'pred_{name.lower()}'] = pred
        cls_pred.append(tmp)
    cls_join = cls_pred[0]
    for t in cls_pred[1:]:
        cls_join = cls_join.merge(t, on=['stock_id','target_month'], how='inner')
    cls_join['date'] = cls_join['target_month'].apply(lambda p: f"{p.year:04d}-{p.month:02d}-28")
    out_cls_tuned = cls_join.rename(columns={'target_month':'month_id'})
    out_cls_tuned['month_id'] = out_cls_tuned['month_id'].apply(lambda p: _format_month(p))
    if full_mode:
        out_cols = ['stock_id','month_id','date','proba_lr','pred_lr','proba_rf','pred_rf']
        if 'proba_hgb' in out_cls_tuned.columns:
            out_cols += ['proba_hgb','pred_hgb']
        out_cls_tuned[out_cols].to_csv('predictions_july_cls_tuned.csv', index=False)

    # Archivo simple con mejor modelo
    proba_best = out_cls_tuned[f'proba_{best_name.lower()}']
    pred_best = out_cls_tuned[f'pred_{best_name.lower()}']
    simple = out_cls_tuned[['stock_id','month_id']].copy()
    simple['pred'] = pred_best
    simple['proba'] = proba_best
    if full_mode:
        simple.to_csv('predictions_july_classification.csv', index=False)

    # Predicciones de regresión
    reg_pred = []
    for name, pipe in all_regs.items():
        yhat = pipe.predict(X_pred[feature_cols])
        tmp = X_pred[['stock_id','target_month']].copy()
        tmp[name] = yhat
        reg_pred.append(tmp)
    reg_join = reg_pred[0]
    for t in reg_pred[1:]:
        reg_join = reg_join.merge(t, on=['stock_id','target_month'], how='inner')
    reg_join['mean_pred_excess'] = reg_join[[c for c in reg_join.columns if c not in ['stock_id','target_month']]].mean(axis=1)
    out_reg = reg_join.rename(columns={'target_month':'month_id'})
    out_reg['month_id'] = out_reg['month_id'].apply(lambda p: _format_month(p))
    if full_mode:
        out_reg.to_csv('predictions_july_excess_regression.csv', index=False)

    # Ensamble simple
    # mean_proba: promedio de (LR, RF, HGB si está)
    prob_cols = [c for c in out_cls_tuned.columns if c.startswith('proba_')]
    comb = out_cls_tuned[['stock_id','month_id'] + prob_cols].copy()
    comb['mean_proba'] = comb[prob_cols].mean(axis=1)
    comb = comb[['stock_id','month_id','mean_proba']]
    comb = comb.merge(out_reg[['stock_id','month_id','mean_pred_excess']], on=['stock_id','month_id'], how='inner')
    comb['pred_cls_ens'] = (comb['mean_proba'] >= 0.5).astype(int)
    comb['pred_reg_sign'] = (comb['mean_pred_excess'] > 0).astype(int)
    if full_mode:
        comb.to_csv('predictions_july_combined.csv', index=False)

    # 4) Generar archivo de entrega: testing_targets.csv con predicciones
    # Clasificación (mejor modelo)
    pred_cls_best = best_clf.predict(X_pred[feature_cols]).astype(int)
    # Regresión (mejor modelo por RMSE en args.train_end)
    pred_reg_best = best_reg.predict(X_pred[feature_cols]).astype(float)

    # Cargar plantilla original y rellenar
    submit = pd.read_csv('testing_targets.csv')
    # Asegurar orden de columnas
    cols = list(submit.columns)
    if cols[:2] != ['month_id','stock_id']:
        # intentar normalizar si difiere el orden
        cols = ['month_id','stock_id','outperform_binary','excess_return']
    # Armar dataframe de predicciones con month_id en formato original (underscore)
    # Tomamos month_id desde la propia plantilla para no alterar formato
    pred_df = submit[['month_id','stock_id']].copy()
    pred_df['outperform_binary'] = pred_cls_best
    pred_df['excess_return'] = pred_reg_best
    pred_df.to_csv('testing_targets.csv', index=False)

    if full_mode:
        print('Done. Files written:')
        print(f' - cv_per_cut_f1.csv, cv_average_f1.csv, cv_fixed_f1_{args.train_end}.csv')
        print(' - predictions_july_classification.csv, predictions_july_cls_tuned.csv')
        print(' - predictions_july_excess_regression.csv, predictions_july_combined.csv')
    print('Done. File written: testing_targets.csv (filled)')


if __name__ == '__main__':
    main()
