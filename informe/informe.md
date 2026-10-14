# Agente de Reinforcement Learning para scalping intradía

**Curso:** Aprendizaje por refuerzo — Maestría en Ingeniería y Analítica de Datos
**Integrantes:** _(hasta 4)_
**Fecha:** _(…)_
**Repositorio / código:** _(enlace o carpeta)_ · **Dataset:** _(nombre, activo, frecuencia, rango de fechas)_

> Instrucciones: reemplaza cada bloque `_(…)_` con tu contenido. Los comentarios en *cursiva* guían qué
> debe responder cada sección según el enunciado; bórralos al final. Los números salen de `reports/` y `runs/`.

---

## 1. Introducción y pregunta de investigación
_(Contexto: decidir cuándo y cuánto invertir. Pregunta central: ¿puede un agente RL aprender una estrategia
que supere a estrategias simples en datos no vistos? Aclara por qué elegiste **scalping** (operar en barras
intradía, posiciones de corta duración, sin overnight) y qué implica: muchas decisiones, costos que dominan,
señal muy débil frente al ruido.)_

## 2. Formulación como proceso de decisión
Esquema: `S_t → A_t → R_{t+1}, S_{t+1}`.

### 2.1 Estado: ¿qué observa el agente?
| Variable | Definición (causal, dentro del día) | Por qué podría ser relevante |
|---|---|---|
| `ret_1`, `ret_5`, `ret_15` | retorno log. a 1, 5 y 15 barras | momentum / reversión de corto plazo |
| `dist_sma` | log(precio / SMA de 30 barras) | sobre-extensión respecto a la media reciente |
| `vol` | desv. est. de retornos (20 barras) | riesgo actual y costo relativo de operar |
| `rsi` | RSI de 14 barras, reescalado a [-1, 1] | presión compradora/vendedora reciente |
| `vol_z` | z-score del volumen (30 barras) | actividad anormal / liquidez |
| `tod` | hora del día normalizada [-1, 1] | patrón intradía y cercanía al cierre forzado |
| `posición` | fracción actual invertida | el costo de cambiar depende de dónde se parte |

Responder explícitamente (preguntas 8–11 del enunciado): (8) qué observa exactamente; (9) ventana histórica de cada indicador;
(10) por qué esos indicadores (apóyate en el EDA, §3); (11) cómo se evita usar el futuro: indicadores *trailing* y locales al día,
scaler ajustado solo con train, split por días, y los tests de `tests/test_lookahead.py` (describe qué verifican y que pasan).

### 2.2 Espacio de acciones
`A = {0 %, 50 %, 100 %}` → acción 0/1/2. Responder (preguntas 12–15): qué puede hacer; significado económico de cada acción;
por qué esa granularidad (menos acciones = menos turnover y aprendizaje más estable); qué **no** puede hacer
(posiciones cortas/apalancadas, tamaños intermedios, órdenes límite, múltiples activos).

## 3. Datos y análisis exploratorio
_(Fuente, frecuencia, rango, nº de barras/días; división temporal: train / val / test con fechas exactas.)_
Incluir (de `reports/figures/`) e **interpretar** cada una:
1. evolución del precio · 2. distribución de retornos (colas pesadas) · 3. regímenes de volatilidad ·
4. perfil intradía de volatilidad y volumen · 5. tendencias y caídas (drawdown del precio) ·
6. autocorrelación de retornos · 7. comportamiento de los indicadores (histogramas, correlaciones) ·
8. relación indicador → retorno siguiente.
_(Concluye: qué de esto justifica el estado, el cierre diario, los costos y la recompensa.)_

## 4. Diseño del entorno
_(Interfaz Gymnasium `reset/step`. Documenta el **timing**: decisión al cierre de `t` con información `≤ t`, ejecución a `close[t]`,
retorno `close[t] → close[t+1]`. Fórmula de valor: `V_{t+1} = V_t (1 − c·|Δpos|)(1 + pos·r_{t+1})`. Costos: fee + slippage = _X_ bps por unidad de turnover.
Cierre forzado a fin de día y por qué. Terminación por ruina. Supuestos y limitaciones: ejecución al cierre, sin impacto de mercado, rebalanceo continuo.
Prueba de robustez con `execution_lag=1`.)_

## 5. Diseño e iteración de la recompensa
Seguir el ciclo del enunciado y mostrar **evidencia** en cada paso:

| Versión | Definición | Problema observado | Cambio |
|---|---|---|---|
| v1 `v1_return` | retorno neto por barra | _(p. ej. opera en exceso / se queda fuera del mercado / …)_ | → v2 |
| v2 `v2_turnover` | log-retorno − λ·turnover | _(…)_ | → v3 |
| v3 `v3_risk` | v2 − μ·aumento de drawdown | _(…)_ | — |

Tabla comparativa en **validación** (`reports/reward_comparison_val.md`) con media ± desv. sobre las semillas, e **interpretación**.
Analizar especialmente: ¿pasa demasiado tiempo fuera del mercado?, ¿opera excesivamente?, ¿solo busca retornos de corto plazo?,
¿asume demasiado riesgo?, ¿explota una debilidad de la recompensa (*reward hacking*)? Incluir al menos un caso concreto.

## 6. Algoritmo y entrenamiento
| Parámetro | Valor |
|---|---|
| Algoritmo | DQN (Double DQN: sí/no) |
| Arquitectura | MLP _(capas)_ |
| Learning rate / batch / γ | _(…)_ |
| Pasos de entrenamiento / episodios | _(…)_ (ventanas de _N_ barras) |
| Replay buffer | _(…)_ |
| Exploración | ε-greedy lineal: _ε_inicial_ → _ε_final_ en _k_ pasos |
| Actualización de la red objetivo | cada _(…)_ pasos |
| Criterio de parada | parada anticipada en validación (paciencia _p_) y mejor checkpoint por _(retorno / calmar)_ |
| Semillas | _(…)_ |
Justifica por qué DQN (acciones discretas) y cómo se eligieron los hiperparámetros **solo con validación**.

## 7. Protocolo de entrenamiento y evaluación
Diagrama train → val → test. Qué se decide con validación (recompensa, hiperparámetros, checkpoint) y qué con test (**solo** la evaluación final,
una vez). Mismas condiciones (período y costos) para agente y baselines. Baselines: aleatorio (media de _N_ semillas), comprar y mantener
(definición del enunciado), B&H intradía y cruce de medias móviles.

## 8. Resultados (held-out)
**Tabla** (de `runs/<run>/eval_test/metrics.md`):

| Estrategia | Retorno acumulado | Máx. drawdown | Costos |
|---|---|---|---|
| Agente RL | | | |
| Comprar y mantener | | | |
| Aleatorio | | | |
| _(B&H intradía, cruce de medias)_ | | | |

Figuras con **interpretación** cada una:
1. curva de capital del agente vs baselines · 2. retorno / drawdown / costos · 3. evolución de posiciones (`positions.png`) ·
4. comportamiento por hora del día · 5. comparación de versiones de recompensa · 6. **al menos un comportamiento inesperado, fallo o anomalía**
(p. ej. colapso a 0 % invertido, rachas de sobre-operación, concentración de operaciones en ciertas horas).
Robustez: sensibilidad a costos (0.5 / 2 / 5 / 10 bps) y `execution_lag=1`.

## 9. Discusión y limitaciones
_(¿Respondiste la pregunta central, incluso si el resultado es negativo? Diferencia entre desempeño en train/val/test. Limitaciones: un solo activo,
período acotado, no estacionariedad, costos supuestos, ejecución al cierre, sin impacto de mercado, varianza entre semillas, riesgo de sobreajuste a validación.
Problemas encontrados y cómo se resolvieron.)_

## 10. Reflexión individual
*(Una por integrante.)*
1. ¿Cuál fue el resultado que más me sorprendió?
2. ¿Qué decisión metodológica tuvo mayor impacto en el agente?
3. ¿Qué cambiaría si repitiera el experimento?
4. ¿Qué comportamiento del agente no logré explicar completamente?
5. ¿Cuál fue la principal diferencia entre lo aprendido en RL y su aplicación a este problema?

## 11. Reproducibilidad
Comandos exactos (copiar del `README.md`), versión de Python/librerías, dataset, semillas y commit/versión del código.
