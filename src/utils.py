import pandas as pd
import numpy as np

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

    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    if rename_cols:
        df = df.rename(columns=rename_cols)

    df.columns = [col.lower().replace('é', 'e').replace('É', 'E') for col in df.columns]

    if 'duracion_recorrido' in df.columns:
        df['duracion_recorrido'] = (
            df['duracion_recorrido']
            .astype(str)
            .str.replace(',', '', regex=False)
        )

    for pair in [('lat_estacion_origen', 'long_estacion_origen'),
                 ('lat_estacion_destino', 'long_estacion_destino')]:
        
        col_lat, col_long = pair
        if col_lat in df.columns and df[col_lat].astype(str).str.contains(',', regex=False).any():
            split_coords = df[col_lat].astype(str).str.split(',', expand=True)
            df[col_lat] = split_coords[0]
            df[col_long] = split_coords[1] if col_long in df.columns else np.nan

    for col in ['lat_estacion_origen', 'long_estacion_origen', 'lat_estacion_destino', 'long_estacion_destino']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')

    df = df.astype({k: v for k, v in dtypes.items() if k in df.columns})

    df = df[[col for col in final_columns if col in df.columns]]
    df = df.loc[:, ~df.columns.duplicated()]

    if 'género' in df.columns and 'genero' in df.columns:
        df['genero'] = df['genero'].fillna(df['género'])
        df = df.drop(columns=['género'])

    df['id_estacion_origen'] = df['id_estacion_origen'].str.replace('BAEcobici', '', regex=False)
    df['id_estacion_destino'] = df['id_estacion_destino'].str.replace('BAEcobici', '', regex=False)
    df['id_recorrido'] = df['id_recorrido'].str.replace('BAEcobici', '', regex=False)
    df['id_usuario'] = df['id_usuario'].str.replace('BAEcobici', '', regex=False)

    if 'duracion_recorrido' in df.columns:
        df['duracion_recorrido'] = pd.to_numeric(df['duracion_recorrido'], errors='coerce')
        df = df[df['duracion_recorrido'] >= 60]

    # delete all rows where id_estacion_origen is NaN
    df = df[df['id_estacion_origen'].notna()]

    return df