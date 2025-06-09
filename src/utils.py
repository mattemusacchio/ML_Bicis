import pandas as pd

dtypes = {
    'id_recorrido': 'string',
    'duracion_recorrido': 'int32',
    'id_estacion_origen': 'string',
    'nombre_estacion_origen': 'string',
    'direccion_estacion_origen': 'string',
    'long_estacion_origen': 'float32',
    'lat_estacion_origen': 'float32',
    'id_estacion_destino': 'string',  # puede tener NaN → usar float
    'nombre_estacion_destino': 'string',
    'direccion_estacion_destino': 'string',
    'long_estacion_destino': 'float32',
    'lat_estacion_destino': 'float32',
    'id_usuario': 'string',  # puede tener decimales o NaN
    'modelo_bicicleta': 'string',
    'genero': 'string'
}

final_columns = list(dtypes.keys()) + ['fecha_origen_recorrido', 'fecha_destino_recorrido']

def clean_and_load(path, drop_cols=None, rename_cols=None):
    df = pd.read_csv(path, parse_dates=['fecha_origen_recorrido', 'fecha_destino_recorrido'], low_memory=False)

    # Eliminar columnas no deseadas
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    # Renombrar columnas si es necesario
    if rename_cols:
        df = df.rename(columns=rename_cols)

    # Normalizar nombres de columnas a minúsculas sin tildes
    df.columns = [col.lower().replace('é', 'e').replace('É', 'E') for col in df.columns]

    # Normalizar duracion_recorrido
    if 'duracion_recorrido' in df.columns:
        df['duracion_recorrido'] = (
            df['duracion_recorrido']
            .astype(str)
            .str.replace(',', '', regex=False)
        )

    import numpy as np

    # Normalizar coordenadas: separar si vienen juntas en un mismo string
    for pair in [('lat_estacion_origen', 'long_estacion_origen'), 
                ('lat_estacion_destino', 'long_estacion_destino')]:
        
        col_lat, col_long = pair
        if col_lat in df.columns and df[col_lat].astype(str).str.contains(',', regex=False).any():
            # Separar si lat y long están juntas
            split_coords = df[col_lat].astype(str).str.split(',', expand=True)
            df[col_lat] = split_coords[0]
            df[col_long] = split_coords[1] if col_long in df.columns else np.nan

    # Reemplazar comas decimales si quedan, y convertir a float
    for col in ['lat_estacion_origen', 'long_estacion_origen', 'lat_estacion_destino', 'long_estacion_destino']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')

    # Forzar tipos
    df = df.astype({k: v for k, v in dtypes.items() if k in df.columns})

    # Reordenar columnas y filtrar solo las necesarias
    df = df[[col for col in final_columns if col in df.columns]]

    # Eliminar columnas duplicadas que pueden haber quedado tras el renombrado
    df = df.loc[:, ~df.columns.duplicated()]

    return df
