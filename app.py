import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
import joblib
from sqlalchemy import create_engine, text
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Pronostico de Precio Ventura",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Personalizado ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    :root {
        --primary-color: #5C1212;
        --sidebar-bg-color: #212529;
        --app-bg-color: #f0f2f5;
        --white: #ffffff;
    }
    body { font-family: 'Roboto', sans-serif; background-color: var(--app-bg-color); }
    .title-banner { background-color: var(--primary-color); color: var(--white); font-size: 2.2em; font-weight: bold; padding: 25px; text-align: center; margin-bottom: 30px; }
    [data-testid="stSidebar"] { background-color: var(--sidebar-bg-color); border-right: 3px solid var(--primary-color); }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label { color: var(--white); }
    [data-testid="stSidebar"] [data-testid="stImage"] { background-color: var(--white); padding: 10px; border-radius: 10px; margin-bottom: 20px; }
    [data-testid="stSidebar"] .stSelectbox > div, [data-testid="stSidebar"] .stMultiSelect > div { background-color: #343a40; color: var(--white); }
    .stButton > button { background-color: var(--primary-color); color: white; border-radius: 6px; border: none; width: 100%; font-weight: 700; padding: 8px 0; }
    .stButton > button:hover { background-color: #4a0f0f; }
    .shiny-fieldset { border: 2px solid var(--primary-color); border-radius: 8px; padding: 15px 20px 20px 20px; margin-bottom: 25px; background-color: var(--white); }
    .shiny-legend { color: var(--primary-color); font-weight: 700; font-size: 1.3rem; padding: 0 10px; margin-left: 15px; width: auto; }
    .metric-box { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem; text-align: center; height: 130px; display: flex; flex-direction: column; justify-content: center; }
    .metric-box h4 { font-size: 0.9rem; color: #6c757d; margin-bottom: 0.5rem; font-weight: 400; }
    .metric-box p { font-size: 1.3rem; color: var(--primary-color); font-weight: 700; margin: 0; }
    .compound-text { font-size: 1.1rem !important; margin: 2px 0 !important; line-height: 1.3 !important; }
    .compound-label { font-size: 0.9rem; color: #343a40; font-weight: 400; margin-right: 8px; }
    .highlight { border: 2px solid var(--primary-color) !important; }
</style>
""", unsafe_allow_html=True)

# --- TEMA PERSONALIZADO PARA GRÁFICAS PLOTLY ---
custom_template = go.layout.Template()
custom_template.layout.colorway = ['#5C1212', '#8E3E3E', '#C87E7E', '#7E9A9A', '#465A5A', '#D2B48C']
pio.templates['ventura_theme'] = custom_template
pio.templates.default = 'ventura_theme'

# --- Banner principal ---
st.markdown('<div class="title-banner">Análisis por Vehiculos Ventura</div>', unsafe_allow_html=True)

# --- FUNCIONES OPTIMIZADAS ---

@st.cache_resource
def cargar_recursos_iniciales():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        def get_path(filename):
            return os.path.join(script_dir, filename)

        with st.spinner("Cargando configuración y filtros... 📊"):
            feature_cols = joblib.load(get_path('feature_cols.joblib'))
            historical_stats = pd.read_parquet(get_path('historical_stats.parquet'))
            opciones_filtros = joblib.load(get_path('opciones_filtros.joblib'))

        with st.spinner("Estableciendo conexión con la base de datos... ☁️"):
            if 'postgres' in st.secrets:
                DATABASE_URL = st.secrets["postgres"]["db_url"]
            else:
                DATABASE_URL = "postgresql://neondb_owner:npg_Rd1vkhV7oCIg@ep-curly-mountain-ae52d80a-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
            engine = create_engine(DATABASE_URL)

        return historical_stats, feature_cols, engine, opciones_filtros
    except Exception as e:
        st.error(f"Error crítico al cargar recursos iniciales: {e}")
        return None, None, None, None

@st.cache_resource
def cargar_modelos():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    def get_path(filename):
        return os.path.join(script_dir, filename)
    
    with st.spinner("Cargando modelos de machine learning... 🧠"):
        modelo_precio = joblib.load(get_path('modelo_precio.joblib'))
        modelo_dias = joblib.load(get_path('modelo_dias.joblib'))
        modelo_recuperacion = joblib.load(get_path('modelo_recuperacion.joblib'))
        modelo_ofertas = joblib.load(get_path('modelo_ofertas.joblib'))
    return modelo_precio, modelo_dias, modelo_recuperacion, modelo_ofertas

@st.cache_data(ttl="1h")
def obtener_datos_filtrados(_engine, filtros_seleccionados):
    where_clauses, params = [], {}
    for key, value in filtros_seleccionados.items():
        if value is not None and value != "TODOS":
            db_col_name = key.upper()
            if key == 'grupo': db_col_name = 'CLIENTE'
            if key == 'año': db_col_name = 'AÑO'
            where_clauses.append(f'"{db_col_name}" = :{key}')
            
            if isinstance(value, np.integer):
                params[key] = int(value)
            else:
                params[key] = value

    query = "SELECT * FROM ventas_historicas"
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    with _engine.connect() as connection:
        df = pd.read_sql_query(text(query), connection, params=params)

    if not df.empty:
        if 'CLIENTE' in df.columns:
            df.rename(columns={'CLIENTE': 'GRUPO'}, inplace=True)
        for col in ['FECHA_DE_INGRESO', 'FECHA_DE_PAGO', 'FECHA_DE_SUBASTA']:
            if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def display_historical_analysis(df):
    st.markdown('<fieldset class="shiny-fieldset"><legend class="shiny-legend">Análisis de Selección</legend>', unsafe_allow_html=True)
    if not df.empty and 'CANTIDAD_OFERTADA' in df.columns and pd.api.types.is_numeric_dtype(df['CANTIDAD_OFERTADA']):
        max_venta = df['CANTIDAD_OFERTADA'].max()
        mean_venta = df['CANTIDAD_OFERTADA'].mean()
        min_venta = df['CANTIDAD_OFERTADA'].min()
        mode_venta = df['CANTIDAD_OFERTADA'].mode().iloc[0] if not df['CANTIDAD_OFERTADA'].value_counts().empty and df['CANTIDAD_OFERTADA'].value_counts().iloc[0] > 1 else df['CANTIDAD_OFERTADA'].median()
        max_reserva = df['PRECIO_RESERVA'].max()
        mean_reserva = df['PRECIO_RESERVA'].mean()
        min_reserva = df['PRECIO_RESERVA'].min()
        mode_reserva = df['PRECIO_RESERVA'].mode().iloc[0] if not df['PRECIO_RESERVA'].value_counts().empty and df['PRECIO_RESERVA'].value_counts().iloc[0] > 1 else df['PRECIO_RESERVA'].median()
        recuperacion_venta_mean = df['RECUPERACION_PRECIO'].mean()
        recuperacion_valor_mean = df['RECUPERACION_VALOR'].mean()
        costo_cliente_mean = df['COSTO_CLIENTE'].mean()
        mean_mercado = df['PRECIO_DE_MERCADO'].mean()
        mean_subastas = df['NUMERO_DE_SUBASTAS'].mean()
        mean_ofertas = df['NUMERO_DE_OFERTAS'].mean()
        mean_dias = df['DIAS_HABILES_VENTA'].mean()
        count_vehiculos = len(df)
    else:
        max_venta, mean_venta, min_venta, mode_venta, max_reserva, mean_reserva, min_reserva, mode_reserva, recuperacion_venta_mean, recuperacion_valor_mean, costo_cliente_mean, mean_mercado, mean_subastas, mean_ofertas, mean_dias, count_vehiculos = [0]*16

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-box"><h4>Precio Máximo Venta</h4><p>${max_venta:,.2f}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box"><h4>Precio de Venta</h4><p class="compound-text"><span class="compound-label">Promedio:</span>${mean_venta:,.2f}</p><p class="compound-text"><span class="compound-label">Moda:</span>${mode_venta:,.2f}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box"><h4>Precio Mínimo Venta</h4><p>${min_venta:,.2f}</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><h4>Máx. Precio Reserva</h4><p>${max_reserva:,.2f}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box"><h4>Precio de Reserva</h4><p class="compound-text"><span class="compound-label">Promedio:</span>${mean_reserva:,.2f}</p><p class="compound-text"><span class="compound-label">Moda:</span>${mode_reserva:,.2f}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box"><h4>Mín. Precio Reserva</h4><p>${min_reserva:,.2f}</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><h4>Recuperaciones Promedio</h4><p class="compound-text"><span class="compound-label">Venta:</span>{recuperacion_venta_mean:.2%}</p><p class="compound-text"><span class="compound-label">Valor:</span>{recuperacion_valor_mean:.2%}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box"><h4>Costo Cliente Promedio</h4><p>${costo_cliente_mean:,.2f}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box"><h4>Valor de Mercado Promedio</h4><p>${mean_mercado:,.2f}</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-box"><h4>Actividad Promedio</h4><p class="compound-text"><span class="compound-label">Subastas:</span>{mean_subastas:.1f}</p><p class="compound-text"><span class="compound-label">Ofertas:</span>{mean_ofertas:.1f}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box"><h4>Días Hábiles de Venta Promedio</h4><p>{mean_dias:.1f}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box highlight"><h4>Vehículos en Selección</h4><p>{count_vehiculos}</p></div>', unsafe_allow_html=True)
    st.markdown('</fieldset>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def generate_comparison_plots(_dataframe, real_col, pred_value, _pipeline, _instance_to_predict, title, format_str):
    real_data = _dataframe[real_col].dropna()
    if real_data.empty:
        st.warning(f"No hay datos históricos suficientes para '{title}' con los filtros actuales.")
        return

    pred_label = f"Predicción: {format_str.format(pred_value)}"
    plot_tabs = st.tabs(["📊 Distribuciones", "🎯 Intervalo y Grupos"])

    with plot_tabs[0]:
        g_c1, g_c2 = st.columns(2)
        with g_c1:
            st.markdown("<h6></h6>", unsafe_allow_html=True)
            fig_violin = px.violin(real_data, y=real_col, box=True, points="all", title=f"Distribución Detallada de {title}")
            fig_violin.add_trace(go.Scatter(x=[title], y=[pred_value], mode='markers', marker=dict(color='red', size=12, symbol='star'), name='Predicción'))
            st.plotly_chart(fig_violin.update_layout(height=400, margin=dict(t=40, b=10, l=10, r=10)), use_container_width=True)
        with g_c2:
            st.markdown("<h6></h6>", unsafe_allow_html=True)
            fig_hist = px.histogram(real_data, x=real_col, title=f"Frecuencia de {title}")
            fig_hist.add_vline(x=pred_value, line_width=3, line_dash="dash", line_color="red", annotation_text=pred_label, annotation_position="top right")
            st.plotly_chart(fig_hist.update_layout(height=400, margin=dict(t=40, b=10, l=10, r=10)), use_container_width=True)
    
    with plot_tabs[1]:
        g_c3, g_c4 = st.columns(2)
        with g_c3:
            st.markdown("<h6></h6>", unsafe_allow_html=True)
            try:
                transformed_instance = _pipeline.named_steps['preprocesador'].transform(_instance_to_predict)
                model = _pipeline.named_steps['regresor']
                predictions_per_tree = [tree.predict(transformed_instance)[0] for tree in model.estimators_]
                lower_bound, upper_bound = np.percentile(predictions_per_tree, [2.5, 97.5])
                fig_interval = px.histogram(real_data, x=real_col, title=f"Incertidumbre del Modelo para {title}")
                fig_interval.add_vrect(x0=lower_bound, x1=upper_bound, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Intervalo 95%", annotation_position="top left")
                fig_interval.add_vline(x=pred_value, line_width=3, line_dash="dash", line_color="red", annotation_text=pred_label, annotation_position="top right")
                st.plotly_chart(fig_interval.update_layout(height=400, margin=dict(t=40, b=10, l=10, r=10)), use_container_width=True)
            except Exception as e: st.warning(f"No se pudo calcular el intervalo de predicción. Error: {e}")
        with g_c4:
            st.markdown("<h6></h6>", unsafe_allow_html=True)
            try:
                stats = _dataframe.groupby('AÑO')[real_col].agg(['mean', 'std']).reset_index().dropna()
                fig_dots = go.Figure(go.Scatter(x=stats['AÑO'], y=stats['mean'], mode='markers', error_y=dict(type='data', array=stats['std'], visible=True), name='Promedio por Año'))
                fig_dots.add_hline(y=pred_value, line_width=3, line_dash="dash", line_color="red", annotation_text=pred_label, annotation_position="bottom right")
                st.plotly_chart(fig_dots.update_layout(title=f"Promedio de {title} por Año", height=400, margin=dict(t=40, b=10, l=10, r=10), xaxis_title="Año del Modelo"), use_container_width=True)
            except Exception: st.warning("No se pudo generar el gráfico de promedios por año.")

# --- Carga de Recursos Iniciales ---
recursos_iniciales = cargar_recursos_iniciales()

if any(res is None for res in recursos_iniciales):
    st.error("La aplicación no pudo cargar su configuración inicial.")
    st.stop()

historical_stats, feature_cols, engine, opciones_filtros = recursos_iniciales

# --- INICIALIZAR SESSION STATE ---
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'df_filtrado' not in st.session_state:
    st.session_state.df_filtrado = pd.DataFrame()

# --- INTERFAZ DE USUARIO (SIDEBAR y LOGO) ---
st.sidebar.image("data:image/png;base64,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", use_container_width=True)
st.sidebar.header("Filtros de análisis")

if st.sidebar.button("Eliminar Filtros"):
    st.session_state.analysis_run = False
    keys_to_reset = [key for key in st.session_state if key.endswith('_widget')]
    for key in keys_to_reset:
        st.session_state[key] = 'TODOS'
    st.rerun()

marca_sel = st.sidebar.selectbox("Marca", ["TODOS"] + opciones_filtros['marcas'], key='marca_widget')
modelos_disponibles = opciones_filtros['modelos_por_marca'].get(marca_sel, []) if marca_sel != 'TODOS' else opciones_filtros['modelos']
modelo_sel = st.sidebar.selectbox("Modelo", ["TODOS"] + sorted(modelos_disponibles), key='modelo_widget')
condicion_sel = st.sidebar.selectbox("Condición de Venta", ["TODOS"] + opciones_filtros['condiciones'])
año_sel = st.sidebar.selectbox("Año del Modelo", ["TODOS"] + opciones_filtros['anios'])
st.sidebar.markdown("---")
clas_venta_sel = st.sidebar.selectbox("Clasificación de Venta", ["TODOS"] + opciones_filtros['clas_venta'])
clas_modelo_sel = st.sidebar.selectbox("Clasificación de Modelo", ["TODOS"] + opciones_filtros['clas_modelo'])
origen_marca_sel = st.sidebar.selectbox("Origen de Marca", ["TODOS"] + opciones_filtros['origen_marca'])
grupo_sel = st.sidebar.selectbox("Grupo", ["TODOS"] + opciones_filtros['clientes'])
segmento_sel = st.sidebar.selectbox("Segmento", ["TODOS"] + opciones_filtros['segmentos'])
combustible_sel = st.sidebar.selectbox("Combustible", ["TODOS"] + opciones_filtros['combustibles'])
st.sidebar.markdown("---")
st.sidebar.header("Filtros por Fecha de Subasta")
año_subasta_sel = st.sidebar.selectbox("Año de Subasta", ["TODOS"] + opciones_filtros['anios_subasta'])
meses_dict = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
meses_nombres_disponibles = [meses_dict[m] for m in sorted(opciones_filtros['meses'])]
mes_subasta_sel_nombre = st.sidebar.selectbox("Mes de Subasta", ["TODOS"] + meses_nombres_disponibles)
mes_subasta_sel = next((k for k, v in meses_dict.items() if v == mes_subasta_sel_nombre), "TODOS")

if st.sidebar.button("Analizar / Estimar"):
    filtros_activos = {
        'marca': marca_sel, 'modelo': modelo_sel, 'condicion_de_venta': condicion_sel,
        'año': año_sel, 'clasificacion_venta': clas_venta_sel, 'clasificacion_modelo': clas_modelo_sel,
        'origen_marca': origen_marca_sel, 'grupo': grupo_sel, 'segmento': segmento_sel,
        'combustible': combustible_sel, 'año_subasta': año_subasta_sel, 'mes_subasta': mes_subasta_sel
    }
    
    with st.spinner("Obteniendo y procesando datos... ⏳"):
        df_filtrado_raw = obtener_datos_filtrados(engine, filtros_activos)
        
        if not df_filtrado_raw.empty:
            df_filtrado = pd.merge(df_filtrado_raw, historical_stats, on=['MARCA', 'MODELO', 'VERSION'], how='left')
        else:
            df_filtrado = df_filtrado_raw

        st.session_state.df_filtrado = df_filtrado
        st.session_state.analysis_run = True
        st.rerun()

# --- PESTAÑAS Y VISUALIZACIONES ---
tab1, tab2, tab3 = st.tabs(["Pronóstico de Resultados", "Análisis Historico", "Evolución de Indicadores"])

with tab1:
    if st.session_state.analysis_run:
        df_filtrado = st.session_state.df_filtrado
        if df_filtrado.empty:
            st.warning("No se encontraron datos para los filtros seleccionados.")
        else:
            modelo_precio, modelo_dias, modelo_recuperacion, modelo_ofertas = cargar_modelos()

            st.markdown('<fieldset class="shiny-fieldset"><legend class="shiny-legend">Pronostico de Indicadores</legend>', unsafe_allow_html=True)
            
            input_data = {}
            marca_pred = df_filtrado['MARCA'].mode().get(0, df_filtrado['MARCA'].iloc[0])
            modelo_pred = df_filtrado['MODELO'].mode().get(0, df_filtrado['MODELO'].iloc[0])
            version_pred = df_filtrado['VERSION'].mode().get(0, df_filtrado['VERSION'].iloc[0])
            
            input_data.update({'MARCA': marca_pred, 'MODELO': modelo_pred, 'VERSION': version_pred})
            
            stats_vehiculo = historical_stats[
                (historical_stats['MARCA'] == marca_pred) & 
                (historical_stats['MODELO'] == modelo_pred) & 
                (historical_stats['VERSION'] == version_pred)
            ]
            
            if not stats_vehiculo.empty:
                input_data.update(stats_vehiculo.iloc[0].to_dict())
            else:
                for col in [c for c in historical_stats.columns if c not in ['MARCA', 'MODELO', 'VERSION']]:
                    input_data[col] = df_filtrado[col].mean()

            input_data.update({
                'AÑO': df_filtrado['AÑO'].mode().get(0, df_filtrado['AÑO'].iloc[0]), 
                'CLASIFICACION_VENTA': df_filtrado['CLASIFICACION_VENTA'].mode().get(0, df_filtrado['CLASIFICACION_VENTA'].iloc[0]),
                'CLASIFICACION_MODELO': df_filtrado['CLASIFICACION_MODELO'].mode().get(0, df_filtrado['CLASIFICACION_MODELO'].iloc[0]), 
                'ORIGEN_MARCA': df_filtrado['ORIGEN_MARCA'].mode().get(0, df_filtrado['ORIGEN_MARCA'].iloc[0]),
                'GRUPO': df_filtrado['GRUPO'].mode().get(0, df_filtrado['GRUPO'].iloc[0]), 
                'SEGMENTO': df_filtrado['SEGMENTO'].mode().get(0, df_filtrado['SEGMENTO'].iloc[0]),
                'COMBUSTIBLE': df_filtrado['COMBUSTIBLE'].mode().get(0, df_filtrado['COMBUSTIBLE'].iloc[0])
            })

            for col in ['NUMERO_DE_SUBASTAS', 'PRECIO_RESERVA', 'COSTO_CLIENTE', 'PRECIO_DE_MERCADO']:
                if col in df_filtrado.columns: input_data[col] = df_filtrado[col].mean()
            
            df_para_predecir = pd.DataFrame([input_data]).reindex(columns=feature_cols)

            if 'GRUPO' in df_para_predecir.columns:
                df_para_predecir.rename(columns={'GRUPO': 'CLIENTE'}, inplace=True)

            columnas_categoricas = [
                'MARCA', 'MODELO', 'VERSION', 'CLASIFICACION_VENTA', 'CLASIFICACION_MODELO', 
                'ORIGEN_MARCA', 'CLIENTE', 'SEGMENTO', 'COMBUSTIBLE'
            ]
            
            for col in columnas_categoricas:
                if col in df_para_predecir.columns:
                    df_para_predecir[col] = df_para_predecir[col].astype(str).fillna('N/A')

            columnas_numericas = [col for col in feature_cols if col not in columnas_categoricas]
            for col in columnas_numericas:
                if col in df_para_predecir.columns:
                    df_para_predecir[col] = pd.to_numeric(df_para_predecir[col], errors='coerce')
                    if df_para_predecir[col].isnull().any():
                        fallback_value = df_filtrado[col].mean()
                        df_para_predecir[col] = df_para_predecir[col].fillna(fallback_value)
            
            pred_precio = modelo_precio.predict(df_para_predecir)[0]
            pred_dias = modelo_dias.predict(df_para_predecir)[0]
            pred_recuperacion = modelo_recuperacion.predict(df_para_predecir)[0]
            pred_ofertas = modelo_ofertas.predict(df_para_predecir)[0]
            
            p1, p2, p3, p4 = st.columns(4)
            with p1: st.markdown(f'<div class="metric-box highlight"><h4>Precio de Venta Estimado</h4><p>${pred_precio:,.2f}</p></div>', unsafe_allow_html=True)
            with p2: st.markdown(f'<div class="metric-box highlight"><h4>Días de Venta Estimados</h4><p>{pred_dias:.1f}</p></div>', unsafe_allow_html=True)
            with p3: st.markdown(f'<div class="metric-box highlight"><h4>Recuperación Estimada</h4><p>{pred_recuperacion:.2%}</p></div>', unsafe_allow_html=True)
            with p4: st.markdown(f'<div class="metric-box highlight"><h4># Ofertas Estimadas</h4><p>{pred_ofertas:.1f}</p></div>', unsafe_allow_html=True)
            st.markdown('</fieldset>', unsafe_allow_html=True)
            
            st.markdown(f"<h3 style='text-align: center; margin-top: 20px;'>Resultados para la Selección</h3>", unsafe_allow_html=True)
            display_historical_analysis(df_filtrado)
            st.markdown('<fieldset class="shiny-fieldset"><legend class="shiny-legend">Visualización Comparativa de la Estimación</legend>', unsafe_allow_html=True)
            v_tab1, v_tab2, v_tab3, v_tab4 = st.tabs(["Precio de Venta", "Días de Venta", "% Recuperación", "# Ofertas"])
            with v_tab1: generate_comparison_plots(df_filtrado, 'CANTIDAD_OFERTADA', pred_precio, modelo_precio, df_para_predecir, "Precio de Venta", "${:,.2f}")
            with v_tab2: generate_comparison_plots(df_filtrado, 'DIAS_HABILES_VENTA', pred_dias, modelo_dias, df_para_predecir, "Días de Venta", "{:.1f}")
            with v_tab3: generate_comparison_plots(df_filtrado, 'RECUPERACION_PRECIO', pred_recuperacion, modelo_recuperacion, df_para_predecir, "% Recuperación", "{:.2%}")
            with v_tab4: generate_comparison_plots(df_filtrado, 'NUMERO_DE_OFERTAS', pred_ofertas, modelo_ofertas, df_para_predecir, "# de Ofertas", "{:.1f}")
            st.markdown('</fieldset>', unsafe_allow_html=True)

            with st.expander("Ver Evaluación de Rendimiento del Modelo (Datos Filtrados)", expanded=True):
                df_evaluacion = df_filtrado.copy()
                if 'GRUPO' in df_evaluacion.columns:
                    df_evaluacion.rename(columns={'GRUPO': 'CLIENTE'}, inplace=True)
                
                X_filtrado = df_evaluacion[feature_cols].copy()
                    
                for col in columnas_categoricas:
                    if col in X_filtrado.columns: X_filtrado[col] = X_filtrado[col].astype(str).fillna('N/A')
                
                for col in columnas_numericas:
                    if col in X_filtrado.columns:
                        if X_filtrado[col].isnull().any():
                            X_filtrado[col] = X_filtrado[col].fillna(df_filtrado[col].mean())
                
                df_filtrado.loc[:, 'PRED_PRECIO'] = modelo_precio.predict(X_filtrado)
                df_filtrado.loc[:, 'PRED_DIAS'] = modelo_dias.predict(X_filtrado)
                df_filtrado.loc[:, 'PRED_RECUPERACION'] = modelo_recuperacion.predict(X_filtrado)
                df_filtrado.loc[:, 'PRED_OFERTAS'] = modelo_ofertas.predict(X_filtrado)
                
                m_tab1, m_tab2, m_tab3, m_tab4 = st.tabs(["Evaluación: Precio", "Evaluación: Días", "Evaluación: Recuperación", "Evaluación: Ofertas"])
                def show_metrics_and_plot(tab, true_col, pred_col, title, format_str, format_str_delta):
                    with tab:
                        temp_df = df_filtrado[[true_col, pred_col]].dropna()
                        if temp_df.empty:
                            st.warning(f"No hay suficientes datos para calcular las métricas de {title}.")
                            return
                        
                        mae = mean_absolute_error(temp_df[true_col], temp_df[pred_col])
                        rmse = np.sqrt(mean_squared_error(temp_df[true_col], temp_df[pred_col]))
                        r2 = r2_score(temp_df[true_col], temp_df[pred_col])
                        st.markdown(f'##### Métricas de Rendimiento: {title}')
                        m1, m2, m3 = st.columns(3)
                        m1.metric(label="Error Absoluto Medio (MAE)", value=format_str.format(mae))
                        m2.metric(label="Raíz Error Cuadrático Medio (RMSE)", value=format_str_delta.format(rmse))
                        m3.metric(label="Coeficiente de Determinación (R²)", value=f"{r2:.2%}")
                        st.markdown("---")
                        fig = px.scatter(temp_df, x=true_col, y=pred_col, title=f'Predicción vs. Realidad para {title}', labels={'x': 'Valor Real', 'y': 'Valor Predicho'})
                        min_val, max_val = min(temp_df[true_col].min(), temp_df[pred_col].min()), max(temp_df[true_col].max(), temp_df[pred_col].max())
                        fig.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color='red', dash='dash'))
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("---")
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Histogram(x=temp_df[true_col], name='Valores Reales', opacity=0.75))
                        fig_dist.add_trace(go.Histogram(x=temp_df[pred_col], name='Valores Predichos', opacity=0.75))
                        fig_dist.update_layout(barmode='overlay', title_text=f'Distribución Comparativa para {title}', xaxis_title_text='Valor', yaxis_title_text='Frecuencia')
                        st.plotly_chart(fig_dist, use_container_width=True)
                        st.markdown("---")
                        temp_df['RESIDUOS'] = temp_df[true_col] - temp_df[pred_col]
                        fig_res = px.histogram(temp_df, x='RESIDUOS', marginal="box", title=f'Distribución de Residuos (Errores) para {title}')
                        fig_res.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_res.update_layout(xaxis_title_text='Error de Predicción (Real - Predicho)', yaxis_title_text='Frecuencia'), use_container_width=True)

                show_metrics_and_plot(m_tab1, 'CANTIDAD_OFERTADA', 'PRED_PRECIO', 'Precio de Venta', "${:,.2f}", "${:,.2f}")
                show_metrics_and_plot(m_tab2, 'DIAS_HABILES_VENTA', 'PRED_DIAS', 'Días de Venta', "{:.1f} días", "{:.1f} días")
                show_metrics_and_plot(m_tab3, 'RECUPERACION_PRECIO', 'PRED_RECUPERACION', '% Recuperación', "{:.2%}", "{:.2%}")
                show_metrics_and_plot(m_tab4, 'NUMERO_DE_OFERTAS', 'PRED_OFERTAS', '# de Ofertas', "{:.1f}", "{:.1f}")
    else:
        st.info("Utiliza los filtros de la barra lateral y presiona 'Analizar / Estimar' para ver los resultados.")

with tab2:
    if st.session_state.analysis_run:
        df_filtrado_final = st.session_state.df_filtrado
        if df_filtrado_final.empty:
            st.warning("No hay datos para mostrar en el análisis de vehículo.")
        else:
            display_historical_analysis(df_filtrado_final)
            st.markdown('<fieldset class="shiny-fieldset"><legend class="shiny-legend">Gráficas Comparativas</legend>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(px.histogram(df_filtrado_final, x='CANTIDAD_OFERTADA', nbins=30, title="1. Distribución de Precio de Venta").update_layout(yaxis_title="Frecuencia", xaxis_title="Precio de Venta"), use_container_width=True)
                st.plotly_chart(px.histogram(df_filtrado_final, x='AÑO', title="3. Distribución de Año-Modelo").update_layout(yaxis_title="Frecuencia", xaxis_title="Año del Modelo", xaxis={'categoryorder':'total descending'}), use_container_width=True)
            with col2:
                st.plotly_chart(px.scatter(df_filtrado_final, x='FECHA_DE_PAGO', y='CANTIDAD_OFERTADA', title="2. Histórico de Precios", hover_data=['MARCA', 'MODELO']).update_layout(xaxis_title="Fecha de Venta", yaxis_title="Precio de Venta"), use_container_width=True)
                st.plotly_chart(px.scatter(df_filtrado_final, x='DIAS_HABILES_VENTA', y='CANTIDAD_OFERTADA', title="4. Precio de Venta vs. Días para Venta", hover_data=['MARCA', 'MODELO']).update_layout(xaxis_title="Días Hábiles para Venta", yaxis_title="Precio de Venta"), use_container_width=True)
            st.markdown('</fieldset>', unsafe_allow_html=True)
            st.markdown('<fieldset class="shiny-fieldset"><legend class="shiny-legend">Análisis de Modelos</legend>', unsafe_allow_html=True)
            if not df_filtrado_final.empty:
                st.write("##### Jerarquía de Volumen de Ventas por Marca y Modelo")
                treemap_data = df_filtrado_final.groupby(['MARCA', 'MODELO']).agg(CANTIDAD=('MODELO', 'size'), PRECIO_PROMEDIO=('CANTIDAD_OFERTADA', 'mean')).reset_index()
                fig_treemap = px.treemap(treemap_data, path=[px.Constant("Marcas"), 'MARCA', 'MODELO'], values='CANTIDAD', title="Volumen de ventas por marca y modelo", custom_data=['PRECIO_PROMEDIO'])
                fig_treemap.update_traces(textinfo="label+percent parent", hovertemplate="<b>%{label}</b><br>Cantidad: %{value}<br>Porcentaje de la Marca: %{percentParent:.2%}<br>Porcentaje del Total: %{percentRoot:.2%}<br>Precio Promedio: $%{customdata[0]:,.2f}")
                st.plotly_chart(fig_treemap.update_layout(margin=dict(t=50, l=25, r=25, b=25), height=700), use_container_width=True)
                if len(df_filtrado_final['MARCA'].unique()) > 1:
                    st.write("##### Rango de Precios de Venta por Marca (Puntos por Modelo)")
                    st.plotly_chart(px.scatter(df_filtrado_final, x='MARCA', y='CANTIDAD_OFERTADA', hover_data=['MODELO', 'VERSION', 'AÑO']).update_layout(xaxis_title=None, yaxis_title="Precio de Venta"), use_container_width=True)
                st.write("##### Volumen de Venta por Marca y Modelo")
                volume_table = df_filtrado_final.groupby(['MARCA', 'MODELO']).agg(CANTIDAD_DE_VEHICULOS=('MODELO', 'size'), PRECIO_PROMEDIO_DE_VENTA=('CANTIDAD_OFERTADA', 'mean'), MODA_DEL_PRECIO_DE_VENTA=('CANTIDAD_OFERTADA', lambda x: x.mode().get(0, np.nan)), PRECIO_RESERVA_PROMEDIO=('PRECIO_RESERVA', 'mean'), MODA_PRECIO_RESERVA=('PRECIO_RESERVA', lambda x: x.mode().get(0, np.nan)), RECUPERACION_PRECIO_PROM=('RECUPERACION_PRECIO', 'mean'), RECUPERACION_VALOR_PROM=('RECUPERACION_VALOR', 'mean'), DIAS_HABILES_PROMEDIO=('DIAS_HABILES_VENTA', 'mean')).sort_values(by='CANTIDAD_DE_VEHICULOS', ascending=False).reset_index()
                volume_table_display = volume_table.copy()
                volume_table_display.columns = [col.replace('_', ' ').title() for col in volume_table.columns]
                format_volume = {'Precio Promedio De Venta': '${:,.2f}', 'Moda Del Precio De Venta': '${:,.2f}', 'Precio Reserva Promedio': '${:,.2f}', 'Moda Precio Reserva': '${:,.2f}', 'Recuperacion Precio Prom': '{:.2%}', 'Recuperacion Valor Prom': '{:.2%}', 'Dias Habiles Promedio': '{:.1f}'}
                st.dataframe(volume_table_display.style.format(format_volume).set_table_styles([{'selector': 'thead th', 'props': [('background-color', '#5C1212'), ('color', 'white')]}]), use_container_width=True)
            st.markdown('</fieldset>', unsafe_allow_html=True)
            st.markdown('<fieldset class="shiny-fieldset"><legend class="shiny-legend">Detalle de Datos Históricos</legend>', unsafe_allow_html=True)
            columnas_display = ['MARCA', 'MODELO', 'VERSION', 'AÑO', 'CONDICION_DE_VENTA', 'COMBUSTIBLE', 'CANTIDAD_OFERTADA', 'COSTO_CLIENTE', 'PRECIO_DE_MERCADO', 'RECUPERACION_PRECIO', 'RECUPERACION_VALOR', 'FECHA_DE_SUBASTA', 'DIAS_HABILES_VENTA', 'NUMERO_DE_SUBASTAS', 'NUMERO_DE_OFERTAS', 'GRUPO', 'SEGMENTO']
            detalle_df_display = df_filtrado_final[[col for col in columnas_display if col in df_filtrado_final.columns]].copy()

            # --- CORRECCIÓN PARA TABLA GIGANTE ---
            if len(detalle_df_display) > 1000:
                st.warning(f"⚠️ Se encontraron {len(detalle_df_display)} registros. Mostrando solo los primeros 1000 para mantener el rendimiento.")
                df_a_mostrar = detalle_df_display.head(1000)
            else:
                df_a_mostrar = detalle_df_display
            
            df_a_mostrar.columns = [col.replace('_', ' ').title() for col in df_a_mostrar.columns]
            format_detalle = {'Cantidad Ofertada': '${:,.2f}', 'Costo Cliente': '${:,.2f}', 'Precio De Mercado': '${:,.2f}', 'Recuperacion Precio': '{:.2%}', 'Recuperacion Valor': '{:.2%}', 'Fecha De Subasta': '{:%Y-%m-%d}', 'Año': '{:.0f}'}
            st.dataframe(df_a_mostrar.style.format(format_detalle).set_table_styles([{'selector': 'thead th', 'props': [('background-color', '#5C1212'), ('color', 'white')]}]), use_container_width=True)
            st.markdown('</fieldset>', unsafe_allow_html=True)
    else:
        st.info("Selecciona filtros y ejecuta el análisis para ver los detalles del vehículo.")

with tab3:
    if st.session_state.analysis_run:
        df_hist = st.session_state.df_filtrado
        if df_hist.empty:
            st.warning("No hay datos para mostrar con los filtros seleccionados.")
        else:
            st.markdown("## Evolución Histórica de Comportamiento")
            st.info("Esta página muestra la evolución mensual del promedio y la moda para los datos filtrados.")
            df_hist['MES_AÑO_SUBASTA'] = df_hist['FECHA_DE_SUBASTA'].dt.to_period('M').dt.to_timestamp()
            cols_to_analyze = {'CANTIDAD_OFERTADA': 'Precio de Venta', 'PRECIO_RESERVA': 'Precio de Reserva', 'COSTO_CLIENTE': 'Costo Cliente', 'PRECIO_DE_MERCADO': 'Valor de Mercado', 'DIAS_HABILES_VENTA': 'Días de Venta', 'NUMERO_DE_OFERTAS': 'Número de Ofertas', 'RECUPERACION_PRECIO': 'Recuperación de Venta', 'RECUPERACION_VALOR': 'Recuperación de Mercado'}
            try:
                with st.spinner("Calculando tendencias históricas..."):
                    st.markdown("### Ventas Totales por Mes (Según Filtros)")
                    df_counts = df_hist.groupby('MES_AÑO_SUBASTA').size().reset_index(name='CONTEO')
                    fig_counts = px.bar(df_counts, x='MES_AÑO_SUBASTA', y='CONTEO', title="Ventas Totales por Mes")
                    st.plotly_chart(fig_counts.update_layout(yaxis_title="Número de Vehículos Vendidos", xaxis_title="Mes de Subasta"), use_container_width=True)
                    st.markdown("---")
                    df_evolucion = df_hist.groupby('MES_AÑO_SUBASTA').agg(**{f'{col}_mean': (col, 'mean') for col in cols_to_analyze.keys()}, **{f'{col}_mode': (col, lambda x: x.mode().get(0, np.nan)) for col in cols_to_analyze.keys()}).reset_index()
                    for col, title in cols_to_analyze.items():
                        st.markdown(f"### {title}")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_evolucion['MES_AÑO_SUBASTA'], y=df_evolucion[f'{col}_mean'], mode='lines+markers', name='Promedio', line=dict(color='#5C1212')))
                        fig.add_trace(go.Scatter(x=df_evolucion['MES_AÑO_SUBASTA'], y=df_evolucion[f'{col}_mode'], mode='lines+markers', name='Moda', line=dict(color='#C87E7E', dash='dash')))
                        fig.update_layout(title=f"Evolución Mensual del Promedio y Moda de {title}", xaxis_title="Mes de Subasta", yaxis_title="Valor", legend_title="Métrica")
                        if 'PRECIO' in col or 'CANTIDAD' in col or 'COSTO' in col or 'VALOR' in col and 'RECUPERACION' not in col: fig.update_layout(yaxis_tickprefix='$', yaxis_tickformat=',.2f')
                        if 'RECUPERACION' in col: fig.update_layout(yaxis_tickformat='.0%')
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("---")
            except Exception as e:
                st.error(f"Ocurrió un error al generar los gráficos de evolución: {e}")
    else:
        st.info("Selecciona filtros y ejecuta el análisis para ver la evolución histórica.")