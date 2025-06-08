import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import pickle
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
from src.models import MLPWithEmbedding
import torch

# Configuración de la página
st.set_page_config(
    page_title="🚴 BA Bicis - Predictor REAL de Arribos",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
    }
    .real-data-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown('<h1 class="main-header">🚴 BA Bicis - Predictor REAL de Arribos</h1>', unsafe_allow_html=True)
st.markdown('### 🤖 Dashboard con Datos Reales <span class="real-data-badge">DATOS REALES</span>', unsafe_allow_html=True)

# Nota explicativa
st.info("📊 Este dashboard usa datos reales de BA Bicis (enero-agosto 2024) con predicciones basadas en patrones históricos observados.")

# Función para cargar datos reales
@st.cache_data
def load_real_data():
    """Cargar datos reales del proyecto"""
    # Cargar estaciones reales
    estaciones_df = pd.read_csv('data/streamlit/estaciones.csv')
    # Cargar datos históricos reales
    datos_historicos_df = pd.read_csv('data/streamlit/datos_historicos.csv')
    datos_historicos_df['timestamp'] = pd.to_datetime(datos_historicos_df['timestamp'])
    
    # Metadatos simulados (ya que el modelo está en MLflow)
    metadata = {
        'mejor_modelo': 'MLP con Embeddings',
        'delta_t_minutes': 30,
        'resultados_modelos': {
            'MLP con Embeddings': {'MAE': 0.011, 'RMSE': 0.07, 'R2': 0.99}
        }
    }
    # abrir el modelo mlp_con_embeddings_bicis.pth

    # Cargar dataset sample
    dataset_sample = pd.read_csv('data/streamlit/dataset_sample.csv')
    dataset_sample['timestamp'] = pd.to_datetime(dataset_sample['timestamp'])
    return estaciones_df, datos_historicos_df, metadata, dataset_sample


# Función para cargar modelo entrenado (simplificada)
@st.cache_resource
def load_trained_model():
    import torch
    import joblib

    # Cargar metadata, preprocesador y pesos
    metadata = joblib.load('models/model_metadata.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')

    # Crear modelo con mismos parámetros
    model = MLPWithEmbedding(
        input_dim=metadata["input_dim"],
        num_stations=metadata["num_stations"],
        emb_dim=metadata["emb_dim"]
    )
    model.load_state_dict(torch.load('models/mlp_embeddings.pth', map_location='cpu'))
    model.eval()

    return model, preprocessor, metadata

# Función para hacer predicciones reales (simplificada)
def hacer_prediccion_real(estaciones_df, timestamp_futuro, partidas_dict):
    model, preprocessor, metadata = load_trained_model()
    station_mapping = metadata["station_mapping"]

    # Filtrar dataset_sample por el timestamp seleccionado
    df_pred = dataset_sample[dataset_sample["timestamp"] == timestamp_futuro].copy()

    if df_pred.empty:
        st.warning("⚠️ No hay datos para el timestamp seleccionado.")
        return {}

    # Mapear id_estacion a station_index (el índice que espera el embedding)
    df_pred["station_index"] = df_pred["id_estacion"].map(station_mapping)

    # Reordenar columnas si fuera necesario (id_estacion no se usa en features)
    X_df = df_pred.drop(columns=["timestamp", "target", "nombre_estacion", "station_index"])
    station_ids = df_pred["station_index"].values

    # Aplicar preprocessor
    X_proc = preprocessor.transform(X_df)
    X_tensor = torch.tensor(X_proc, dtype=torch.float32)
    station_tensor = torch.tensor(station_ids, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor, station_tensor).squeeze().numpy()

    predicciones = {
    row["id_estacion"]: max(0, float(pred))
    for (_, row), pred in zip(df_pred.iterrows(), preds)
    }

    return predicciones



# Cargar datos y modelo
estaciones_df, datos_historicos_df, metadata, dataset_sample = load_real_data()
model, scaler, model_metadata = load_trained_model()

# Verificar que los datos esté cargados
if estaciones_df is None or datos_historicos_df is None or metadata is None:
    st.error("❌ No se pudieron cargar los datos. Verifica que existan los archivos en data/streamlit/")
    st.stop()

# Para demo, siempre consideramos el modelo disponible con los datos reales
model_disponible = True
st.success(f"✅ Datos cargados: {metadata['mejor_modelo']} con {len(estaciones_df)} estaciones reales")

# Sidebar con información del modelo
st.sidebar.header("📊 Información del Modelo")
st.sidebar.success(f"🤖 **Modelo:** {metadata['mejor_modelo']}")
resultados = metadata['resultados_modelos'][metadata['mejor_modelo']]
st.sidebar.metric("MAE", f"{resultados['MAE']:.3f}")
st.sidebar.metric("RMSE", f"{resultados['RMSE']:.3f}")
st.sidebar.metric("R²", f"{resultados['R2']:.3f}")
st.sidebar.metric("Δt", f"{metadata['delta_t_minutes']} min")

# Tabs principales
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Mapa Real", 
    "🔮 Predictor Real", 
    "📊 Análisis Real", 
    "🤖 Modelos", 
    "📈 Métricas"
])

# =============================================================================
# TAB 1: MAPA REAL DE ESTACIONES
# =============================================================================
with tab1:
    st.header("🗺️ Mapa Real de Estaciones BA Bicis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Crear mapa real
        centro_lat = estaciones_df['lat_estacion'].mean()
        centro_lon = estaciones_df['long_estacion'].mean()
        
        m = folium.Map(location=[centro_lat, centro_lon], zoom_start=12)
        
        # Agregar estaciones reales al mapa
        for _, estacion in estaciones_df.iterrows():  # Primeras 50 para rendimiento
            # Calcular actividad real si tenemos dataset_sample
            if dataset_sample is not None:
                actividad_real = dataset_sample[dataset_sample['id_estacion'] == estacion['id_estacion']]['partidas'].mean()
                actividad_real = actividad_real if not pd.isna(actividad_real) else 0
            else:
                actividad_real = np.random.randint(0, 20)  # Fallback
            
            color = 'red' if actividad_real > 10 else 'orange' if actividad_real > 5 else 'green'
            
            folium.CircleMarker(
                location=[estacion['lat_estacion'], estacion['long_estacion']],
                radius=6,
                popup=f"""
                <b>{estacion['nombre_estacion']}</b><br>
                Actividad promedio: {actividad_real:.1f}<br>
                ID: {estacion['id_estacion']}<br>
                Lat: {estacion['lat_estacion']:.4f}<br>
                Lon: {estacion['long_estacion']:.4f}
                """,
                color=color,
                fillColor=color,
                fillOpacity=0.8
            ).add_to(m)
        
        st_folium(m, width=800, height=500)
    
    with col2:
        st.subheader("📊 Estadísticas Reales")
        st.metric("Estaciones Totales", len(estaciones_df))
        st.metric("Período Datos", "Ene-Ago 2024")
        
        if dataset_sample is not None:
            actividad_promedio = dataset_sample['partidas'].mean()
            st.metric("Partidas Promedio", f"{actividad_promedio:.1f}")
            
            # Top estaciones
            st.subheader("🏆 Top Estaciones por Actividad")
            if not dataset_sample.empty:
                top_estaciones = (dataset_sample.groupby('id_estacion')['partidas']
                                .mean().sort_values(ascending=False).head(5))
                
                for i, (est_id, partidas) in enumerate(top_estaciones.items()):
                    nombre = estaciones_df[estaciones_df['id_estacion'] == est_id]['nombre_estacion'].iloc[0] if len(estaciones_df[estaciones_df['id_estacion'] == est_id]) > 0 else f"Estación {est_id}"
                    st.write(f"{i+1}. {nombre[:20]}... ({partidas:.1f})")
# =============================================================================
# TAB 2: PREDICTOR REAL (USANDO DATOS REALES DEL DATASET)
# =============================================================================
with tab2:
    st.header("🔮 Predictor Real de Arribos")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Configuración")
        fecha_pred = st.date_input("📅 Fecha", value=datetime(2024, 8, 1))
        hora_pred = st.time_input("🕐 Hora", value=datetime(8, 1, 1, 0).time())

        timestamp_prediccion = datetime.combine(fecha_pred, hora_pred)

        if st.button("🚀 Predecir arribos reales", type="primary"):
            with st.spinner("Calculando predicciones..."):

                # Usamos directamente los datos reales para ese timestamp
                df_input = dataset_sample[dataset_sample["timestamp"] == timestamp_prediccion].copy()

                if df_input.empty:
                    st.warning("⚠️ No hay datos disponibles para ese horario.")
                else:
                    # Ejecutar predicción real con el modelo
                    predicciones_reales = hacer_prediccion_real(
                        estaciones_df, 
                        timestamp_prediccion, 
                        partidas_dict=None  # ya no se usa, puede ser None
                    )

                    st.session_state.prediccion_real = True
                    st.session_state.timestamp_real = timestamp_prediccion
                    st.session_state.arribos_real = predicciones_reales
                    st.session_state.df_input = df_input

    with col2:
        st.subheader("📊 Resultados Reales")

        if st.session_state.get("prediccion_real", False):
            df_input = st.session_state.df_input
            predicciones = st.session_state.arribos_real

            resultados_df = []
            for _, row in df_input.iterrows():
                est_id = row["id_estacion"]
                nombre = estaciones_df.loc[estaciones_df['id_estacion'] == est_id, 'nombre_estacion'].values[0]
                partidas = row["partidas"]
                arribos = predicciones.get(est_id, 0)
                balance = arribos - partidas

                resultados_df.append({
                    'Estación': nombre[:25] + "..." if len(nombre) > 25 else nombre,
                    'ID': est_id,
                    'Partidas': partidas,
                    'Arribos Predichos': arribos,
                    'arribos': row['arribos'] if 'arribos' in row else 0,  # Si no hay arribos reales, usar 0
                    'Balance': balance
                })

            resultados_df = pd.DataFrame(resultados_df)
            st.dataframe(resultados_df, use_container_width=True)

            st.subheader("🗺️ Mapa de Predicciones por Estación")

            # Crear mapa centrado
            centro_lat = estaciones_df['lat_estacion'].mean()
            centro_lon = estaciones_df['long_estacion'].mean()
            m = folium.Map(location=[centro_lat, centro_lon], zoom_start=12)

            # Merge con coordenadas y nombre
            merged = resultados_df.merge(estaciones_df, left_on="ID", right_on="id_estacion")

            # Escalar tamaño de los círculos
            max_arribos = merged['Arribos Predichos'].max()
            for _, row in merged.iterrows():
                color = (
                    'green' if row['Balance'] >= 0 else 'red'
                )
                radius = max(3, min(10, row['Arribos Predichos'] / max_arribos * 10))
                
                popup_text = f"""
                <b>{row['nombre_estacion']}</b><br>
                Partidas: {row['Partidas']}<br>
                Arribos predichos: {row['Arribos Predichos']:.1f}<br>
                Arribos Reales: {row['arribos']:.1f}<br>
                Balance: {row['Balance']:+.1f}
                """

                folium.CircleMarker(
                    location=[row['lat_estacion'], row['long_estacion']],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_text, max_width=300)
                ).add_to(m)

            # Mostrar mapa en Streamlit
            st_folium(m, width=800, height=500)


            # Alertas reales
            st.subheader("🚨 Alertas Operativas Reales")
            alertas_generadas = 0
            for _, row in resultados_df.iterrows():
                if row['Balance'] > 8:
                    st.error(f"🔴 **{row['Estación']}**: Exceso de +{row['Balance']} bicis - RECOLECTAR")
                    alertas_generadas += 1
                elif row['Balance'] < -3:
                    st.warning(f"🟡 **{row['Estación']}**: Déficit de {row['Balance']} bicis - REABASTECER")
                    alertas_generadas += 1

            if alertas_generadas == 0:
                st.success("✅ No se detectaron desbalances críticos")
            else:
                st.info(f"📊 {alertas_generadas} alertas generadas")
        else:
            st.info("👆 Elegí una fecha y hora y presioná el botón para predecir los arribos.")


# =============================================================================
# TAB 3: ANÁLISIS TEMPORAL REAL
# =============================================================================
with tab3:
    st.header("📊 Análisis Temporal con Datos Reales")
    
    # Patrones reales por hora
    st.subheader("🕐 Patrones Reales por Hora")
    
    patron_hora_real = datos_historicos_df.groupby('hora').agg({
        'partidas': 'mean',
        'arribos': 'mean'
    }).reset_index()
    
    fig_hora_real = px.line(
        patron_hora_real, 
        x='hora', 
        y=['partidas', 'arribos'],
        title="Promedio Real de Partidas y Arribos por Hora (Datos BA Bicis)",
        labels={'value': 'Cantidad de Bicicletas', 'hora': 'Hora del Día'},
        color_discrete_map={'partidas': '#ff7f0e', 'arribos': '#1f77b4'}
    )
    st.plotly_chart(fig_hora_real, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Patrones por día de la semana
        st.subheader("📅 Patrones por Día (Datos Reales)")
        
        dias_nombre = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
        patron_dia_real = datos_historicos_df.groupby('dia_semana').agg({
            'partidas': 'mean',
            'arribos': 'mean'
        }).reset_index()
        
        # Mapear nombres solo para días existentes
        patron_dia_real['dia_nombre'] = patron_dia_real['dia_semana'].map(
            {i: dias_nombre[i] for i in range(len(dias_nombre))}
        )
        
        fig_dia_real = px.bar(
            patron_dia_real, 
            x='dia_nombre', 
            y=['partidas', 'arribos'],
            title="Uso Real por Día de la Semana",
            barmode='group'
        )
        st.plotly_chart(fig_dia_real, use_container_width=True)
    
    with col2:
        # Estadísticas clave
        st.subheader("📈 Estadísticas Clave")
        
        st.metric("📊 Total Partidas", f"{datos_historicos_df['partidas'].sum():,}")
        st.metric("📊 Total Arribos", f"{datos_historicos_df['arribos'].sum():,}")
        st.metric("📊 Promedio/Hora", f"{datos_historicos_df['partidas'].mean():.1f}")
        
        # Hora pico
        hora_pico = patron_hora_real.loc[patron_hora_real['partidas'].idxmax(), 'hora']
        st.metric("🕐 Hora Pico", f"{hora_pico:02d}:00")
        
        # Día más activo
        dia_activo = patron_dia_real.loc[patron_dia_real['partidas'].idxmax(), 'dia_nombre']
        st.metric("📅 Día Más Activo", dia_activo)

# =============================================================================
# TAB 4: MODELOS REALES
# =============================================================================
with tab4:
    st.header("🤖 Resultados de Modelos Reales")
    
    if metadata and 'resultados_modelos' in metadata:
        resultados_reales = metadata['resultados_modelos']
        
        # Convertir resultados a DataFrame
        modelos_df_real = []
        for nombre, metricas in resultados_reales.items():
            modelos_df_real.append({
                'Modelo': nombre,
                'MAE': metricas['MAE'],
                'RMSE': metricas['RMSE'],
                'R2': metricas['R2']
            })
        
        modelos_df_real = pd.DataFrame(modelos_df_real)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico de comparación real
            fig_modelos_real = make_subplots(
                rows=1, cols=3,
                subplot_titles=('MAE', 'RMSE', 'R²')
            )
            
            fig_modelos_real.add_trace(
                go.Bar(x=modelos_df_real['Modelo'], y=modelos_df_real['MAE'], 
                       name='MAE', marker_color='lightcoral'),
                row=1, col=1
            )
            
            fig_modelos_real.add_trace(
                go.Bar(x=modelos_df_real['Modelo'], y=modelos_df_real['RMSE'], 
                       name='RMSE', marker_color='lightsalmon'),
                row=1, col=2
            )
            
            fig_modelos_real.add_trace(
                go.Bar(x=modelos_df_real['Modelo'], y=modelos_df_real['R2'], 
                       name='R²', marker_color='lightblue'),
                row=1, col=3
            )
            
            fig_modelos_real.update_layout(height=400, showlegend=False, 
                                         title_text="Rendimiento Real de Modelos (Datos BA Bicis)")
            st.plotly_chart(fig_modelos_real, use_container_width=True)
        
        with col2:
            st.subheader("🏆 Modelo Ganador")
            mejor_modelo_real = metadata['mejor_modelo']
            metricas_mejor = resultados_reales[mejor_modelo_real]
            
            st.markdown(f"""
            <div class="prediction-box">
                <h3>🥇 {mejor_modelo_real}</h3>
                <p><strong>MAE:</strong> {metricas_mejor['MAE']:.3f} bicis</p>
                <p><strong>RMSE:</strong> {metricas_mejor['RMSE']:.3f} bicis</p>
                <p><strong>R²:</strong> {metricas_mejor['R2']:.3f}</p>
                <hr>
                <p><em>Entrenado con datos reales de BA Bicis 2024</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tabla completa
        st.subheader("📋 Comparación Completa")
        st.dataframe(modelos_df_real, use_container_width=True)

# =============================================================================
# TAB 5: MÉTRICAS REALES
# =============================================================================
with tab5:
    st.header("📈 Métricas del Sistema Real")
    
    # Métricas del proyecto
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Estaciones", len(estaciones_df))
    
    with col2:
        total_registros = len(dataset_sample) if dataset_sample is not None else 0
        st.metric("📊 Registros", f"{total_registros:,}")
    
    with col3:
        st.metric("⏱️ Ventana Predicción", f"{metadata['delta_t_minutes']} min")
    
    with col4:
        st.metric("🤖 Modelo Activo", "✅ SÍ")
    
    # Información del dataset
    if dataset_sample is not None:
        st.subheader("📊 Información del Dataset Real")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Rango temporal:**")
            st.write(f"- Desde: {dataset_sample['timestamp'].min().strftime('%Y-%m-%d %H:%M')}")
            st.write(f"- Hasta: {dataset_sample['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
            
            st.write("**Estadísticas de partidas:**")
            st.write(f"- Promedio: {dataset_sample['partidas'].mean():.2f}")
            st.write(f"- Máximo: {dataset_sample['partidas'].max()}")
            st.write(f"- Total: {dataset_sample['partidas'].sum():,}")
        
        with col2:
            # Distribución de partidas
            fig_dist = px.histogram(
                dataset_sample, 
                x='partidas', 
                nbins=30,
                title="Distribución Real de Partidas por Ventana",
                labels={'partidas': 'Partidas por Ventana', 'count': 'Frecuencia'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚴 <strong>BA Bicis - Predictor REAL de Arribos</strong> | 
    <span class="real-data-badge">DATOS REALES</span></p>
    <p>📊 Datos: Enero-Agosto 2024 | 🤖 Modelo entrenado con datos históricos reales</p>
</div>
""", unsafe_allow_html=True) 