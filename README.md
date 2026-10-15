# Agente de Reinforcement Learning para scalping intradía

Proyecto de la actividad evaluativa de *Aprendizaje por refuerzo*. Un agente **DQN** aprende, con velas intradía,
qué fracción de su capital exponer a un activo (0 %, 50 % o 100 %), pagando costos de transacción y cerrando
posiciones al final de cada sesión (scalping, sin posiciones overnight). Se compara contra baselines simples bajo
exactamente las mismas condiciones y se evalúa en un período *held-out* que no se usa para tomar ninguna decisión.

**Pregunta central:** ¿puede un agente de RL aprender una estrategia de inversión que se comporte mejor que estrategias
simples cuando se enfrenta a datos que no ha visto durante su entrenamiento?

**Producto:** código reproducible del agente y del entorno de simulación (este repositorio) e informe
(`informe/informe.md`) que documenta: (a) la formulación del problema, (b) el diseño del entorno y de la función de
recompensa, (c) el protocolo de evaluación, (d) la comparación contra baselines y (e) la discusión y reflexión finales.

> **Sobre los datos:** sin `data.path`, el código genera una serie **sintética** solo para comprobar que todo corre;
> sus resultados no son evidencia sobre mercados reales. Para el trabajo se usa el dataset del profesor
> (o, como respaldo para pruebas, SPY a 5 minutos con `yfinance`; ver [Datos](#datos)).

## Contenido

1. [Paso 1 · Estado y acciones](#paso-1--formular-el-estado-y-las-acciones)
2. [Paso 2 · Entorno de simulación](#paso-2--entorno-de-simulación-con-costo-por-operar)
3. [Paso 3 · Función de recompensa](#paso-3--diseñar-e-iterar-la-función-de-recompensa)
4. [Paso 4 · Entrenamiento (DQN)](#paso-4--entrenar-el-agente-dqn)
5. [Paso 5 · Evaluación held-out vs baselines](#paso-5--evaluar-en-datos-no-vistos-contra-baselines)
6. [Paso 6 · Discusión y reflexión](#paso-6--discutir-y-reflexionar)
7. [Paso 7 · Presentación](#paso-7--presentar-el-proyecto)
8. [Instalación, datos y reproducibilidad](#instalación-datos-y-reproducibilidad) · [Google Colab](#ejecución-en-google-colab) · [Estructura](#estructura-del-repositorio)

---

## Paso 1 · Formular el estado y las acciones

**Estado** (`src/data.py`). En cada barra `t` el agente observa 8 indicadores derivados del precio, más su posición actual.
Todos se calculan hacia atrás (*trailing*) y **dentro de cada día**, de modo que nunca mezclan sesiones ni miran barras futuras.

| Variable | Definición | Por qué podría ser relevante |
|---|---|---|
| `ret_1`, `ret_5`, `ret_15` | retorno logarítmico a 1, 5 y 15 barras | momentum o reversión de corto plazo |
| `dist_sma` | log(precio / media móvil de 30 barras) | sobre-extensión respecto a la media reciente |
| `vol` | desviación estándar de retornos (20 barras) | riesgo actual y costo relativo de operar |
| `rsi` | RSI de 14 barras, reescalado a [-1, 1] | presión compradora o vendedora reciente |
| `vol_z` | z-score del volumen (30 barras) | actividad anormal / liquidez |
| `tod` | hora del día normalizada [-1, 1] | patrón intradía y cercanía al cierre forzado |
| posición | fracción actual invertida | el costo de cambiar depende de dónde se parte |

**Acciones** (`configs/default.yaml → env.action_levels`). `A = {0, 0.5, 1.0}`: mantener 0 %, 50 % o 100 % del capital
invertido. Poca granularidad limita el turnover y facilita el aprendizaje con DQN. Con esta representación el agente
**no** puede vender en corto, apalancarse, usar tamaños intermedios ni operar varios activos. Se puede cambiar el espacio
(p. ej. `[0, 0.25, 0.5, 0.75, 1.0]`) justificándolo en el informe.

**Anti-lookahead.** El escalamiento (media y desviación) se ajusta **solo con el período de entrenamiento** y se guarda
en `scaler.json`; el split es por días completos. `tests/test_lookahead.py` verifica automáticamente que (i) los
indicadores en `t` no cambian si se borra o altera el futuro, (ii) el scaler no depende de validación ni test, y
(iii) los períodos son cronológicos y disjuntos.

## Paso 2 · Entorno de simulación con costo por operar

`src/environment.py` implementa `TradingEnv` con la interfaz Gymnasium (`reset`, `step`):

```python
obs, info = env.reset()
while not done:
    action = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

**Timing (lookahead-safe).** El agente decide al cierre de la barra `t` con información `≤ t`, la orden se ejecuta a
`close[t]` y la ganancia o pérdida del paso es el movimiento `close[t] → close[t+1]`:

```
V_{t+1} = V_t · (1 − c·|p' − p|) · (1 + p'·r_{t+1})        c = (fee_bps + slippage_bps) / 10⁴
```

* **Costo:** `fee_bps + slippage_bps` por unidad de turnover (default 1 + 1 bps; es un supuesto de scalping, no el de
  10 bps del ejemplo del enunciado). Cambiarlo con `--set env.fee_bps=… --set env.slippage_bps=…`.
* **Cierre diario:** en la última barra de cada día la posición se fuerza a 0 (se paga el costo) y no se gana el gap overnight.
* **Rezago de ejecución:** `env.execution_lag=1` ejecuta la orden una barra después (más realista; prueba de robustez).
* **Ruina:** en **entrenamiento** el episodio termina si el capital cae bajo `env.ruin_fraction` del inicial. En
  **evaluación** esa regla se desactiva para que todas las estrategias recorran el período completo.
* `tests/test_env.py` verifica costos, timing, cierre diario, rezago y consistencia de la recompensa.

## Paso 3 · Diseñar e iterar la función de recompensa

Tres versiones, seleccionables con `reward.name` (los retornos por barra son ~1e-4; `reward.scale` los multiplica):

| Versión | Definición | Idea |
|---|---|---|
| `v1_return` | `(V_{t+1} − V_t) / V_t` | retorno neto por barra, incluye costos |
| `v2_turnover` | `log(V_{t+1}/V_t) − λ·turnover` | castiga operar de más (`reward.turnover_penalty`) |
| `v3_risk` | `v2 − μ·máx(0, aumento de drawdown)` | además castiga profundizar el drawdown (`reward.drawdown_penalty`) |

Ciclo de iteración: entrenar → observar el comportamiento → identificar problemas (¿opera en exceso?, ¿se queda fuera
del mercado?, ¿asume demasiado riesgo?, ¿*reward hacking*?) → modificar → reentrenar → comparar en **validación**:

```bash
python -m src.experiments --rewards v1_return v2_turnover v3_risk --seeds 0 1 2
# -> runs/<recompensa>_s<semilla>/ y reports/reward_comparison_val.{md,csv,png}
```

**Resultado preliminar** (SPY, velas de 5 min, ~59 días, **validación**, 2 semillas; ver [Datos](#datos)):

| Estrategia (validación) | Retorno acum. | Máx. drawdown | Costos ($) | Nº cambios | % invertido |
|---|---|---|---|---|---|
| RL `v1_return` | −0.45 % ± 0.49 | −0.91 % ± 0.48 | 978 | 70 | 50 % |
| RL `v2_turnover` | 0.14 % ± 0.07 | −0.83 % ± 1.11 | 239 | 16 | 34 % |
| RL `v3_risk` | 0.04 % ± 0.05 | −0.02 % ± 0.02 | 10 | 1 | 0 % |
| Aleatorio (media de 10) | −8.10 % | −8.16 % | 8,021 | 628 | 66 % |
| Comprar y mantener | −0.06 % | −2.04 % | 20 | 1 | 100 % |
| Cruce de medias | −0.22 % | −1.16 % | 801 | 40 | 52 % |

A más penalización, menos operaciones: `v1` opera de más, `v2` equilibra y `v3` es tan pasiva que se queda fuera del
mercado (comportamiento inesperado a analizar). Con ~8 días de validación y 2 semillas las diferencias entre agentes no
son estadísticamente distinguibles de 0; **reemplaza esta tabla con la del dataset oficial**.

## Paso 4 · Entrenar el agente (DQN)

`src/agent.py` (red, replay buffer, Double DQN) y `src/train.py` (bucle de entrenamiento). Hiperparámetros por defecto
(`configs/default.yaml → dqn`; los ajustes por dataset van con `--set`):

| Parámetro | Valor |
|---|---|
| Arquitectura | MLP de 2 capas ocultas `[128, 128]`, ReLU |
| Learning rate · batch · γ | 1e-4 · 128 · 0.99 |
| Pasos · inicio de aprendizaje · frecuencia | 150 000 · 5 000 · cada 4 pasos |
| Replay buffer · actualización de red objetivo | 100 000 · cada 1 000 pasos |
| Exploración | ε-greedy lineal 1.0 → 0.05 durante `eps_decay_fraction` (50 %) de los pasos |
| Episodio de entrenamiento | ventanas aleatorias de `episode_len` barras (780 por defecto) |
| Criterio de parada | parada anticipada por validación (`patience`) |
| Semillas | `seed` en el YAML o `--seed` |

**Selección del checkpoint.** La validación (nunca el test) elige el mejor checkpoint (`dqn.selection_metric`: `return`
o `calmar`), y **solo se consideran evaluaciones posteriores al fin del decaimiento de ε**. Sin esa condición, un
checkpoint temprano casi sin entrenar que se queda 100 % fuera del mercado (retorno 0 %) ganaba por defecto frente a
políticas que operaban y perdían un poco.

```bash
python -m src.train --reward v2_turnover --seed 0 --out runs/v2_turnover_s0
```

## Paso 5 · Evaluar en datos no vistos contra baselines

**Protocolo.** División cronológica por días completos: 60 % entrenamiento · 20 % validación · 20 % test (held-out).
El agente aprende con el entrenamiento; recompensa, hiperparámetros y checkpoint se deciden con la validación; el test
se usa **una sola vez**, al final, con el agente ya definido. `evaluate.py` exige `--final` para tocar el test.

**Baselines** (mismos datos y costos que el agente): agente **aleatorio** (media de `evaluation.random_seeds` semillas),
**comprar y mantener** (una compra al inicio, incluye gaps overnight), **B&H intradía** (100 % invertido en el mismo
entorno que el agente, con cierre diario) y **cruce de medias móviles**.

**Métricas:** retorno acumulado, máximo drawdown, costos de transacción (más Sharpe anualizado, nº de cambios de
posición y % de tiempo invertido), con curva de capital y evolución de posiciones.

```bash
python -m src.evaluate --run runs/v2_turnover_s0 --split val                  # diagnóstico en validación
python -m src.evaluate --run runs/v2_turnover_s0 --split test --final         # held-out: UNA sola vez
```

Genera en `runs/<run>/eval_<split>/`: `metrics.md/csv`, `equity_curves.png`, `positions.png` y
`position_by_time_of_day.png`.

**Robustez** (recomendada en scalping, el costo domina el resultado):

```bash
python -m src.evaluate --run runs/v2_turnover_s0 --split val --cost-bps 0.5
python -m src.evaluate --run runs/v2_turnover_s0 --split val --cost-bps 5
python -m src.evaluate --run runs/v2_turnover_s0 --split val --set env.execution_lag=1
```

## Paso 6 · Discutir y reflexionar

Se documenta en el informe (§9 discusión y limitaciones, §10 reflexión individual). Puntos a cubrir:

* ¿Se respondió la pregunta central, incluso con resultado negativo? Diferencia entre train, validación y test.
* Comportamientos inesperados: p. ej. un agente que se queda fuera del mercado, o cuya validación empeora mientras
  la pérdida de entrenamiento baja (sobreajuste con pocos datos).
* Limitaciones: un solo activo, período acotado, no estacionariedad, costos supuestos, ejecución al cierre, varianza
  entre semillas, riesgo de sobreajustar a la validación.
* Reflexión individual (5 preguntas): resultado sorprendente, decisión metodológica de mayor impacto, qué se cambiaría,
  comportamiento no explicado y diferencia entre lo aprendido en RL y su aplicación aquí.

## Paso 7 · Presentar el proyecto

Material para el panel o foro, todo generado por el código: curvas de capital (`equity_curves.png`), tabla vs baselines
(`metrics.md`), posiciones del agente, comparación de recompensas (`reports/reward_comparison_val.*`), sensibilidad a
costos y un ejemplo de comportamiento inesperado. Lleva la plantilla `informe/informe.md` completa.

---

## Instalación, datos y reproducibilidad

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                 # numpy, pandas, gymnasium, torch, matplotlib, pyyaml, pytest
python -m pytest -q                                     # anti-lookahead + mecánica del entorno
```

### Datos

CSV con columnas `timestamp, open, high, low, close, volume` (se aceptan alias como `datetime`, `date`, `apertura`,
`cierre`). Debe ser **intradía** (1, 5 o 15 min); con datos diarios el código se detiene con un aviso, porque scalping
requiere varias barras por día.

1. Copia el archivo a `data/` y pon `data.path: data/<archivo>.csv` en `configs/default.yaml` (o usa `--set data.path=…`).
2. Ajusta a la frecuencia y sesión del activo: `data.session_start`, `data.session_minutes`,
   `evaluation.bars_per_year` y `dqn.episode_len`. Valores por defecto para velas de **1 min** de una sesión de 390 min
   (252 días × 390 = 98 280). Para velas de **5 min**: `bars_per_year=19656` y `episode_len=156`.
3. Si no se dispone del dataset del profesor, para pruebas se puede descargar SPY a 5 minutos con `yfinance`
   (Yahoo solo entrega ~60 días: sirve para verificar el pipeline, no para una evaluación seria).

### Semillas

`seed` (YAML o `--seed`) fija Python, NumPy y PyTorch; el buffer, la exploración ε y el entorno usan generadores derivados
de esa semilla. El agente aleatorio usa semillas `1000 + i`; los datos sintéticos, `data.synthetic.seed`. En CPU los
resultados son reproducibles con las mismas versiones de librerías. Cualquier valor del YAML se puede sobrescribir con
`--set clave=valor`. Reporta media ± desviación sobre varias semillas.

## Ejecución en Google Colab

```python
# 1) clonar e instalar
!git clone https://github.com/osdato98-sys/invest-app-rl.git
%cd invest-app-rl
!pip install -q gymnasium pytest yfinance
!python -m pytest -q
```
```python
# 2) datos de prueba: SPY a 5 minutos (si ya tienes el CSV del profesor, súbelo a data/ y define DATA)
import yfinance as yf
raw = yf.download("SPY", period="59d", interval="5m", auto_adjust=False, progress=False)
raw.columns = raw.columns.get_level_values(0)
df = raw.reset_index().rename(columns={"Datetime": "timestamp"})[["timestamp","Open","High","Low","Close","Volume"]]
df.columns = ["timestamp","open","high","low","close","volume"]
df.to_csv("data/spy_5m.csv", index=False)

DATA = "data/spy_5m.csv"
PARAMS = ("--set evaluation.bars_per_year=19656 --set dqn.episode_len=156 "
          "--set dqn.total_steps=60000 --set dqn.eval_every=10000 --set dqn.patience=4 "
          "--set dqn.eps_decay_fraction=0.3")
```
```python
# 3) EDA y comparación de recompensas (validación)
!python -m src.eda --set data.path={DATA}
!python -m src.experiments --rewards v1_return v2_turnover v3_risk --seeds 0 1 --set data.path={DATA} {PARAMS}
print(open("reports/reward_comparison_val.md").read())
```
```python
# 4) evaluar un run elegido en validación; el test solo al final (--final)
RUN = "runs/v2_turnover_s0"
!python -m src.evaluate --run {RUN} --split val
# !python -m src.evaluate --run {RUN} --split test --final
```
```python
# 5) guardar resultados antes de que se cierre la sesión (runs/ no se sube a git)
!zip -qr resultados.zip runs reports
from google.colab import files
files.download("resultados.zip")
```

## Estructura del repositorio

```
configs/default.yaml     todos los parámetros (datos, costos, recompensa, DQN, evaluación)
src/data.py              carga, indicadores causales, scaler (solo train), split temporal, generador sintético
src/environment.py       TradingEnv (Gymnasium): timing, costos, cierre diario, 3 recompensas
src/agent.py             DQN / Double DQN, replay buffer, red MLP
src/train.py             entrenamiento; la validación decide checkpoint y parada anticipada
src/baselines.py         aleatorio, comprar y mantener, cruce de medias
src/metrics.py           episodio completo y métricas (retorno, drawdown, costos, Sharpe, ...)
src/evaluate.py          agente + baselines, tablas y gráficos (protege el test con --final)
src/experiments.py       comparación de versiones de recompensa en validación
src/eda.py               análisis exploratorio (solo train + validación)
tests/                   anti-lookahead y mecánica del entorno
notebooks/eda.ipynb      EDA interactivo
informe/informe.md       plantilla del informe con todas las secciones pedidas
```

## Recomendaciones metodológicas

* La rigurosidad metodológica vale más que un retorno alto: un agente que falla en los datos no vistos, pero con una
  evaluación disciplinada, puede obtener la nota máxima.
* **No se hacen ajustes sobre el período de evaluación final.** Recompensa, hiperparámetros y checkpoint se eligen solo
  con validación; el test se evalúa una única vez con el agente ya definido.
* Instrumento de evaluación: rúbrica (ver Estrategia de Evaluación, Rúbrica 4).

## Referencias

* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2.ª ed.). MIT Press.
* Powell, W. B. (2019). *Reinforcement learning and stochastic optimization*. Wiley.
* Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
