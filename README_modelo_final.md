# MODELO M0 - PREDICTOR DE ARRIBOS DE BICICLETAS GCBA

## Descripción

El **Modelo M0** es nuestro modelo final de Machine Learning para predecir la cantidad de arribos de bicicletas en las estaciones del sistema de bicicletas públicas del Gobierno de la Ciudad de Buenos Aires (GCBA).

### Características Técnicas

- **Algoritmo**: XGBoost Regressor
- **Objetivo**: Predicción de arribos de bicicletas (variable continua)
- **Métricas de entrenamiento**:
  - R² en entrenamiento: 0.4641
  - R² en validación: 0.4230
  - MAE en validación: 0.3964
  - MSE en validación: 0.5332

## Archivos del Modelo

```
proyecto_final/
├── modelo_m0_predictor.py          # Script principal ejecutable
├── models/xgb_model_M0.pkl         # Modelo entrenado (50MB)
├── README_MODELO_M0.md             # Esta documentación
└── ejemplo_uso_modelo_m0.py        # Script de ejemplo
```

## Formato de Datos de Entrada

El modelo requiere un archivo CSV con las siguientes columnas:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `timestamp` | datetime | Fecha y hora del registro |
| `id_estacion` | int | Identificador único de la estación |
| `lat_estacion` | float | Latitud de la estación |
| `long_estacion` | float | Longitud de la estación |
| `despachos_count` | int | Cantidad de despachos en el intervalo |
| `edad_usuario_mean` | float | Edad promedio de usuarios |
| `edad_usuario_std` | float | Desviación estándar de edad |
| `proporcion_mujeres` | float | Proporción de mujeres (0-1) |
| `hora` | int | Hora del día (0-23) |
| `dia_semana` | int | Día de la semana (0-6) |
| `es_fin_semana` | int | 1 si es fin de semana, 0 si no |
| `mes` | int | Mes del año (1-12) |
| `dia_mes` | int | Día del mes (1-31) |
| `año` | int | Año |

## Uso del Modelo

### Instalación de Dependencias

```bash
pip install pandas numpy scikit-learn xgboost joblib
```

### Uso Básico

```bash
python modelo_final.py --input datos_test.csv --output predicciones.csv
```

### Opciones Avanzadas

```bash
# Usar un modelo específico
python modelo_final.py --input datos_test.csv --output predicciones.csv --modelo models/xgb_model_M0.pkl

# Evaluar predicciones (si los datos incluyen valores reales)
python modelo_final.py --input datos_con_target.csv --output predicciones.csv --evaluar
```

### Ejemplo de Uso Programático

```python
from modelo_final import ModeloM0Predictor
import pandas as pd

# Cargar datos
df = pd.read_csv('mis_datos.csv')

# Inicializar predictor
predictor = ModeloM0Predictor('models/xgb_model_M0.pkl')

# Hacer predicciones
predicciones = predictor.predecir(df)

# Generar submission
predictor.generar_submission(df, predicciones, 'mis_predicciones.csv')
```

## Formato de Salida

El modelo genera un archivo CSV con las siguientes columnas:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id_estacion` | int | Identificador de la estación |
| `timestamp` | datetime | Fecha y hora del registro |
| `arribos_predichos` | int | Cantidad predicha de arribos |

### Ejemplo de Salida

```csv
id_estacion,timestamp,arribos_predichos
1,2024-09-09 12:00:00,3
1,2024-09-09 12:30:00,5
2,2024-09-09 12:00:00,2
2,2024-09-09 12:30:00,4
```

## Validaciones del Modelo

El modelo incluye las siguientes validaciones automáticas:

1. **Validación de columnas**: Verifica que todas las features requeridas estén presentes
2. **Validación de tipos**: Asegura tipos de datos correctos
3. **Validación de valores nulos**: Maneja valores faltantes
4. **Validación de rango**: Las predicciones son siempre ≥ 0

## Métricas de Evaluación

Si los datos de entrada incluyen la columna `arribos_count` (valores reales), el modelo puede evaluar automáticamente:

- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coeficiente de determinación)

## Limitaciones y Consideraciones

1. **Datos temporales**: El modelo funciona mejor con datos dentro del rango temporal de entrenamiento (2024)
2. **Estaciones nuevas**: Para estaciones no vistas durante el entrenamiento, usar coordenadas y features temporales
3. **Valores extremos**: El modelo puede tener menor precisión en condiciones muy atípicas
4. **Dependencias**: Requiere las mismas features que durante el entrenamiento

## Troubleshooting

### Error: "No se encontró el modelo"
- Verificar que existe `models/xgb_model_M0.pkl`
- Usar la opción `--modelo` para especificar ruta personalizada

### Error: "Columnas faltantes"
- Verificar que el CSV incluye todas las columnas requeridas
- Revisar nombres exactos (case-sensitive)

### Error: "Datos de entrada inválidos"
- Verificar tipos de datos correctos
- Asegurar que no hay valores nulos en columnas críticas

## Contacto

**Autores**: Matteo Musacchio, Tiziano Demarco  
**Proyecto**: Modelado Bicicletas Públicas GCBA - 1er Semestre 2025 