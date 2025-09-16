
Resumen
- Tarea básica: clasificar si cada acción supera al índice US Monash en 2023-07 (métrica: F1).
- Tarea avanzada: predecir el exceso de retorno continuo (métrica: RMSE).
- Validación temporal y prevención de leakage.

Estructura creada
- src/data.py: carga, parsing de fechas, unión de datasets y generación de X/y sin leakage (features de t-1 para predecir t).
- src/models.py: preprocesamiento (ColumnTransformer), modelos de clasificación (LR, RF, HGB) y regresión (Ridge, RF, HGBR), CV temporal, predicciones.
- run_pipeline.py: ejecuta extremo a extremo CV y predicción para 2023-07.
- requirements.txt: dependencias principales.

Entradas esperadas (en el raíz del repo)
- stock_data.csv, monashIndex.csv, company_info.csv
- vix_index.csv, us_10_year_treasury.csv, us_5_year_treasury.csv, fed_inflation_rate.csv, fed_funds_rate.csv, fed_unemployment_rate.csv
- training_targets.csv, testing_targets.csv

Salidas
- cv_per_cut_f1.csv, cv_average_f1.csv, cv_fixed_f1_2023-06.csv
- predictions_july_classification.csv (mejor modelo) y predictions_july_cls_tuned.csv (por modelo)
- predictions_july_excess_regression.csv y predictions_july_combined.csv

Uso rápido
1) Instalar dependencias (opcional si ya las tienes):
   pip install -r requirements.txt

2) Generar solo el archivo de entrega (recomendado):
   python run_pipeline.py

   Esto entrena con semilla fija y escribe únicamente `testing_targets.csv` con las 4 columnas requeridas.

3) (Opcional) Generar métricas y archivos auxiliares:
   python run_pipeline.py --full

   Produce `cv_*.csv` y predicciones intermedias para análisis, además de `testing_targets.csv`.

Notas
- Las features de predicción para el mes t se construyen usando únicamente datos disponibles hasta el fin de t-1 (evita leakage), incluyendo macro de t-1 y rasgos estáticos de la empresa.
- Para predicciones 2023-07, se usan features de 2023-06.
