# RL para scalping intradía (DQN + Gymnasium)

Proyecto de Reinforcement Learning: un agente DQN aprende a decidir cuánto capital exponer a un
activo (0 %, 50 %, 100 %) con velas **intradía**, pagando costos de transacción y cerrando posiciones
al final de cada sesión (sin overnight). Se compara contra baselines bajo exactamente las mismas
condiciones y se evalúa en un período **held-out** que no se usa para tomar decisiones.

> **Sobre los datos incluidos:** sin `data.path`, el código genera una serie **sintética** solo para
> comprobar que todo corre. Los números obtenidos con ella **no son evidencia** sobre mercados reales.
> Para el trabajo, apunta `data.path` al dataset del profesor.

## 1. Instalación

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                 # numpy, pandas, gymnasium, torch, matplotlib, pyyaml, pytest
python -m pytest -q                                     # auditoría anti-lookahead + pruebas del entorno
```

## 2. Dataset

CSV con columnas `timestamp, open, high, low, close, volume` (se aceptan alias como `datetime`, `date`,
`apertura`, `cierre`...). Debe ser **intradía** (1, 5, 15 min); con datos diarios el código se detiene
con un mensaje explicativo, porque scalping no tiene sentido sin varias barras por día.

1. Copia el archivo a `data/` (p. ej. `data/dataset_profesor.csv`).
2. En `configs/default.yaml` pon `data.path: data/dataset_profesor.csv`.
3. Ajusta `data.session_start` y `data.session_minutes` a la sesión del activo
   (p. ej. mercado US: `"09:30"` y `390`; activo 24 h: `"00:00"` y `1440`) y `evaluation.bars_per_year`.

División temporal por días completos: 60 % train / 20 % val / 20 % test (configurable). El **test no se usa**
para elegir hiperparámetros, recompensa ni checkpoint.

## 3. Cómo ejecutar

```bash
# (a) EDA (solo train+val) -> reports/figures/*.png y reports/eda_summary.md
python -m src.eda
#     o abre notebooks/eda.ipynb

# (b) Iteración de la recompensa: entrena v1/v2/v3 x 3 semillas y compara en VALIDACIÓN
python -m src.experiments --rewards v1_return v2_turnover v3_risk --seeds 0 1 2
#     -> runs/<recompensa>_s<semilla>/ y reports/reward_comparison_val.{md,csv,png}

# (c) Entrenar un agente puntual
python -m src.train --reward v3_risk --seed 0 --out runs/v3_risk_s0

# (d) Baselines + agente en validación (diagnóstico)
python -m src.evaluate --run runs/v3_risk_s0 --split val

# (e) EVALUACIÓN FINAL en held-out: una sola vez, con el agente ya elegido en validación
python -m src.evaluate --run runs/v3_risk_s0 --split test --final

# (f) Robustez (opcional, recomendado en scalping): sensibilidad a costos y a rezago de ejecución
python -m src.evaluate --run runs/v3_risk_s0 --split val --cost-bps 0.5
python -m src.evaluate --run runs/v3_risk_s0 --split val --cost-bps 5
python -m src.evaluate --run runs/v3_risk_s0 --split val --set env.execution_lag=1
```

Cualquier valor de `configs/default.yaml` se puede sobrescribir: `--set dqn.lr=0.0003 --set env.fee_bps=2`.
Cada run guarda `config.yaml`, `scaler.json`, `log.csv`, `best.pt` y `final.pt`; la evaluación guarda
`metrics.md/csv`, la curva de capital, las posiciones y el comportamiento por hora del día.

## 4. Reproducibilidad (semillas)

* `seed` en el YAML (o `--seed`) fija Python, NumPy y PyTorch; el buffer, la exploración ε y el entorno
  usan generadores derivados de esa semilla. El agente aleatorio usa semillas `1000 + i` (`evaluation.random_seeds`).
* Los datos sintéticos usan `data.synthetic.seed`.
* En CPU los resultados son reproducibles con la misma versión de librerías; en GPU puede haber pequeñas
  diferencias numéricas. Se recomienda reportar media ± desviación sobre varias semillas.

## 5. Estructura

```
configs/default.yaml     todos los parámetros (datos, costos, recompensa, DQN, evaluación)
src/data.py              carga, indicadores causales, scaler (solo train), split temporal, generador sintético
src/environment.py       TradingEnv (Gymnasium): timing, costos, cierre diario, 3 recompensas
src/agent.py             DQN / Double DQN, replay buffer, red MLP
src/train.py             entrenamiento; validación decide checkpoint y parada anticipada
src/baselines.py         aleatorio, comprar y mantener, cruce de medias
src/metrics.py           episodio completo + métricas (retorno, drawdown, costos, Sharpe, ...)
src/evaluate.py          agente + baselines, tablas y gráficos (protege el test con --final)
src/experiments.py       comparación de versiones de recompensa en validación
src/eda.py               análisis exploratorio (sin tocar test)
tests/                   anti-lookahead y mecánica del entorno
notebooks/eda.ipynb      EDA interactivo
informe/informe.md       plantilla del informe con todas las secciones pedidas
```

## 6. Decisiones de diseño (para justificar en el informe)

* **Timing:** se decide al cierre de la barra `t` con información `<= t` y se gana el retorno `close[t] → close[t+1]`.
  `env.execution_lag=1` retrasa la ejecución una barra (más realista; usar como prueba de robustez).
* **Costos:** `(fee_bps + slippage_bps)` por unidad de turnover; el default (2 bps en total) es un supuesto
  de scalping, **no** el 10 bps del ejemplo del enunciado. Reporta sensibilidad (0.5 / 2 / 5 / 10 bps):
  en scalping el resultado depende muchísimo del costo.
* **Cierre diario:** la posición se fuerza a 0 en la última barra del día. Por eso hay dos "comprar y mantener":
  el del enunciado (una compra, incluye gaps overnight) y el **B&H intradía** (mismo entorno que el agente).
* **Recompensas:** `v1_return` (retorno neto), `v2_turnover` (log-retorno − castigo por turnover),
  `v3_risk` (v2 − castigo por profundizar el drawdown). Los retornos por barra se escalan (`reward.scale`).
* **Ruina:** el entrenamiento termina el episodio si el capital cae bajo `env.ruin_fraction` del inicial; en
  **evaluación** esa regla se desactiva para que todas las estrategias recorran el período completo
  (si no, un agente aleatorio "se detendría" y subestimaría su pérdida).
* **Anti-lookahead:** indicadores trailing y locales al día; scaler ajustado solo con train; verificados en `tests/`.
