# Rediseño de Experimentos 8, 9 y 10: Prueba Sistemática de la Hipótesis de Complejidad

## Objetivo General

Probar sistemáticamente la hipótesis: **"Cuando la física es más compleja, la jaula se rompe"**

Cada experimento compara un sistema físico SIMPLE vs uno COMPLEJO del mismo dominio, midiendo:
1. R² Score (precisión)
2. Cage Status (correlación con variables humanas)
3. Generalización (extrapolación)

---

## EXPERIMENTO 8: Mecánica Clásica Simple vs. Mecánica Cuántica

### Hipótesis
La mecánica cuántica (compleja, contraintuitiva) debería romper la jaula, mientras que la mecánica clásica simple (intuitiva) debería mantenerla bloqueada.

### Diseño

**Parte A: Sistema Simple (Mecánica Clásica)**
- **Dominio**: Oscilador armónico simple 1D
- **Física**: $x(t) = A \cos(\omega t + \phi)$
- **Input**: $[A, \omega, \phi, t]$
- **Output**: Posición $x(t)$
- **Complejidad**: Baja - ecuación explícita, intuitiva
- **Predicción**: 🔒 **CAGE LOCKED** - reconstruye $A$, $\omega$, $\phi$

**Parte B: Sistema Complejo (Mecánica Cuántica)**
- **Dominio**: Partícula en pozo cuántico 1D
- **Física**: $\psi(x) = \sqrt{\frac{2}{L}} \sin\left(\frac{n\pi x}{L}\right)$ (estado estacionario)
- **Input**: $[n, L, x]$ (número cuántico, ancho del pozo, posición)
- **Output**: Probabilidad de densidad $|\psi(x)|^2$
- **Complejidad**: Alta - cuantización, estados discretos, no intuitivo
- **Predicción**: 🔓 **CAGE BROKEN** - solución distribuida, no reconstruye $n$ explícitamente

### Métricas
- R² Score en ambas partes
- Cage Analysis: correlación con variables humanas ($A$, $\omega$ vs $n$, $L$)
- Extrapolación: fuera del rango de entrenamiento

---

## EXPERIMENTO 9: Sistemas Lineales vs. No Lineales (Caos Determinístico)

### Hipótesis
Los sistemas no lineales caóticos (complejos, impredecibles) deberían romper la jaula, mientras que los sistemas lineales (simples, predecibles) deberían mantenerla bloqueada.

### Diseño

**Parte A: Sistema Simple (Lineal)**
- **Dominio**: Circuito RLC en serie (oscilador amortiguado)
- **Física**: $Q(t) = Q_0 e^{-\gamma t} \cos(\omega_d t + \phi)$
- **Input**: $[Q_0, \gamma, \omega_d, \phi, t]$
- **Output**: Carga $Q(t)$
- **Complejidad**: Baja - solución analítica explícita
- **Predicción**: 🔒 **CAGE LOCKED** - reconstruye parámetros

**Parte B: Sistema Complejo (No Lineal Caótico)**
- **Dominio**: Atractor de Lorenz (sistema caótico)
- **Física**: Ecuaciones diferenciales acopladas no lineales:
  - $\dot{x} = \sigma(y - x)$
  - $\dot{y} = x(\rho - z) - y$
  - $\dot{z} = xy - \beta z$
- **Input**: $[x_0, y_0, z_0, t]$ (condiciones iniciales, tiempo)
- **Output**: $x(t)$ (una coordenada del atractor)
- **Complejidad**: Alta - caos determinístico, sensibilidad a condiciones iniciales
- **Predicción**: 🔓 **CAGE BROKEN** - no puede reconstruir parámetros del sistema

### Métricas
- R² Score en ambas partes
- Cage Analysis: correlación con parámetros del sistema
- Sensibilidad: pequeñas variaciones en condiciones iniciales

---

## EXPERIMENTO 10: Baja vs. Alta Dimensionalidad (Sistemas de Muchos Cuerpos)

### Hipótesis
Los sistemas de muchos cuerpos (alta dimensionalidad, complejos) deberían romper la jaula, mientras que los sistemas de pocos cuerpos (baja dimensionalidad, simples) deberían mantenerla bloqueada.

### Diseño

**Parte A: Sistema Simple (Baja Dimensionalidad)**
- **Dominio**: Problema de 2 cuerpos (órbitas keplerianas)
- **Física**: Órbita elíptica: $r(\theta) = \frac{a(1-e^2)}{1+e\cos(\theta)}$
- **Input**: $[a, e, \theta]$ (semieje mayor, excentricidad, ángulo)
- **Output**: Distancia radial $r(\theta)$
- **Complejidad**: Baja - solución analítica, 2 cuerpos
- **Predicción**: 🔒 **CAGE LOCKED** - reconstruye $a$, $e$

**Parte B: Sistema Complejo (Alta Dimensionalidad)**
- **Dominio**: Sistema de N cuerpos gravitacionales (N=5-10)
- **Física**: Integración numérica de ecuaciones de movimiento:
  - $\ddot{\vec{r}}_i = G \sum_{j \neq i} \frac{m_j (\vec{r}_j - \vec{r}_i)}{|\vec{r}_j - \vec{r}_i|^3}$
- **Input**: $[\vec{r}_1, \vec{r}_2, ..., \vec{r}_N, \vec{v}_1, ..., \vec{v}_N, t]$ (posiciones, velocidades, tiempo)
- **Output**: Energía total del sistema $E(t)$
- **Complejidad**: Alta - no hay solución analítica, comportamiento caótico
- **Predicción**: 🔓 **CAGE BROKEN** - no puede reconstruir variables individuales

### Métricas
- R² Score en ambas partes
- Cage Analysis: correlación con variables individuales vs. propiedades emergentes
- Escalabilidad: cómo cambia el rendimiento con N

---

## Metodología Unificada

### 1. Simuladores
- Cada parte (Simple y Compleja) tiene su propio simulador físico
- Mismos rangos de parámetros cuando sea posible
- Mismo tamaño de dataset (2000-3000 muestras)

### 2. Modelos
- **Baseline Darwiniano**: Polynomial Regression (grado 4)
- **Chaos Model**: Optical Chaos (4096 features, FFT mixing)
- Mismos hiperparámetros en ambas partes para comparación justa

### 3. Evaluación
- **R² Score**: Precisión de predicción
- **Cage Analysis**: 
  - Correlación máxima de features internas con variables humanas
  - Si correlación > 0.9: 🔒 CAGE LOCKED
  - Si correlación < 0.3: 🔓 CAGE BROKEN
  - Si 0.3 < correlación < 0.9: 🟡 CAGE UNCLEAR
- **Extrapolación**: Train/test split por rango de parámetros
- **Robustez**: Ruido del 5% en inputs

### 4. Criterios de Éxito

**Hipótesis confirmada si**:
- Parte Simple: 🔒 CAGE LOCKED + R² alto
- Parte Compleja: 🔓 CAGE BROKEN + R² alto
- Diferencia significativa en correlaciones

**Hipótesis refutada si**:
- Ambas partes tienen el mismo cage status
- Parte Compleja tiene peor R² que Simple
- No hay diferencia en correlaciones

---

## Estructura de Archivos

```
experiment_8_classical_vs_quantum/
├── experiment_8_classical_vs_quantum.py
├── benchmark_experiment_8.py
└── README.md

experiment_9_linear_vs_chaos/
├── experiment_9_linear_vs_chaos.py
├── benchmark_experiment_9.py
└── README.md

experiment_10_low_vs_high_dim/
├── experiment_10_low_vs_high_dim.py
├── benchmark_experiment_10.py
└── README.md
```

---

## Análisis Comparativo Final

Al final de los 3 experimentos, crear un análisis comparativo que muestre:

1. **Tabla de Resultados**:
   | Experimento | Parte Simple | Parte Compleja | Diferencia |
   |-------------|--------------|----------------|------------|
   | 8 (Clásico vs Cuántico) | R², Cage | R², Cage | ? |
   | 9 (Lineal vs Caos) | R², Cage | R², Cage | ? |
   | 10 (Baja vs Alta Dim) | R², Cage | R², Cage | ? |

2. **Conclusión General**:
   - ¿Se confirma la hipótesis de complejidad?
   - ¿Hay patrones consistentes?
   - ¿Qué tipo de complejidad rompe la jaula?

---

## Notas de Diseño

- **Imparcialidad**: No diseñar los sistemas para favorecer la hipótesis
- **Rigor**: Validar que ambos sistemas son físicamente correctos
- **Comparabilidad**: Mantener variables de control constantes
- **Transparencia**: Documentar todas las decisiones de diseño

