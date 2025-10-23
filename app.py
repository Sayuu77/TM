import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Configuración de la página
st.set_page_config(
    page_title="Reconocimiento Inteligente de Imágenes",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema moderno con colores azules
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: #e2e8f0;
    }
    .main-title {
        font-size: 2.5rem;
        text-align: center;
        background: linear-gradient(45deg, #3b82f6, #60a5fa, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        font-weight: 800;
    }
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .confidence-high { color: #10b981; font-weight: 700; }
    .confidence-medium { color: #f59e0b; font-weight: 700; }
    .confidence-low { color: #ef4444; font-weight: 700; }
    .stButton button {
        background: linear-gradient(45deg, #3b82f6, #60a5fa);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-title">🤖 Reconocimiento Inteligente de Imágenes</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Teachable Machine & Deep Learning</div>', unsafe_allow_html=True)

# Sidebar con información y controles
with st.sidebar:
    st.markdown("""
        <h2 style="margin: 0">Configuración</h2>
    """, unsafe_allow_html=True)
    
    # Selector de modo
    modo = st.radio(
        "Modo de entrada:",
        ["📷 Cámara Web", "📁 Subir Imagen"],
        index=0
    )

    
    # Configuración de confianza mínima
    confianza_minima = st.slider(
        "Confianza mínima para detección:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Ajusta el nivel de confianza mínimo para considerar una detección válida"
    )
    
    # Instrucciones
    st.markdown("Instrucciones")
    st.markdown("""
    1. Selecciona modo cámara o subir imagen
    2. Ajusta la confianza mínima
    3. Toma foto o sube imagen
    4. Analiza resultados en tiempo real
    """)

# Cargar modelo (con manejo de errores)
@st.cache_resource
def load_ai_model():
    try:
        model = load_model('keras_model.h5')
        return model
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

model = load_ai_model()

# Clases del modelo (personalizar según tu modelo)
CLASSES = [
    "Izquierda 👈",
    "Arriba 👆", 
    "Derecha 👉",
    "Abajo 👇",
    "Centro 🎯"
]

# Función para preprocesar imagen
def preprocess_image(image):
    try:
        # Redimensionar a 224x224
        image = image.resize((224, 224))
        # Convertir a array numpy
        img_array = np.array(image)
        # Normalizar
        normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
        return normalized_image_array
    except Exception as e:
        st.error(f"Error procesando imagen: {e}")
        return None

# Función para realizar predicción
def predict_image(model, image_array):
    try:
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = image_array
        prediction = model.predict(data, verbose=0)
        return prediction[0]
    except Exception as e:
        st.error(f"Error en predicción: {e}")
        return None

# Función para mostrar resultados
def display_results(predictions, confianza_minima):
    # Encontrar la clase con mayor probabilidad
    max_prob = np.max(predictions)
    max_class = np.argmax(predictions)
    
    # Crear DataFrame para resultados
    results_df = pd.DataFrame({
        'Clase': CLASSES[:len(predictions)],
        'Confianza': predictions[:len(CLASSES)]
    })
    
    # Ordenar por confianza
    results_df = results_df.sort_values('Confianza', ascending=False)
    
    # Resultado principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        
        if max_prob >= confianza_minima:
            st.markdown("**Detección Confirmada**")
            
            # Mostrar resultado principal con emoji y color
            if max_prob > 0.8:
                confidence_class = "confidence-high"
                emoji = "🎯"
            elif max_prob > 0.6:
                confidence_class = "confidence-medium" 
                emoji = "📊"
            else:
                confidence_class = "confidence-low"
                emoji = "🔍"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <h1 style="margin: 0; font-size: 2.5rem;">{CLASSES[max_class]}</h1>
                <div class="{confidence_class}" style="font-size: 1.8rem; margin: 0.5rem 0;">
                    {emoji} {max_prob:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown("### ⚠️ **Detección Incierta**")
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <h3 style="color: #f59e0b;">Confianza máxima: {max_prob:.1%}</h3>
                <p>La confianza está por debajo del umbral mínimo ({confianza_minima:.0%})</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Métricas rápidas
        st.markdown("Métricas")
        st.markdown(f"""
        <div class="metric-card">
            <div>Confianza Máxima</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #3b82f6;">
                {max_prob:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div>Clase Detectada</div>
            <div style="font-size: 1.2rem; font-weight: bold; color: #10b981;">
                {CLASSES[max_class].split(' ')[0]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráfico de barras de probabilidades
    st.markdown("###Distribución de Probabilidades")
    fig = px.bar(
        results_df, 
        x='Clase', 
        y='Confianza',
        color='Confianza',
        color_continuous_scale='viridis',
        height=300
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla detallada
    st.markdown("Resultados Detallados")
    
    # Aplicar formato de color a la tabla
    def color_confianza(val):
        if val > 0.8:
            color = '#10b981'
        elif val > 0.6:
            color = '#f59e0b'
        else:
            color = '#ef4444'
        return f'color: {color}; font-weight: 600'
    
    styled_df = results_df.style.format({
        'Confianza': '{:.1%}'
    }).applymap(lambda x: color_confianza(x) if isinstance(x, (int, float)) else '', 
               subset=['Confianza'])
    
    st.dataframe(styled_df, use_container_width=True)
    
    return max_class, max_prob

# Historial de predicciones
if 'history' not in st.session_state:
    st.session_state.history = []

# Contenido principal
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("Captura de Imagen")
    
    if modo == "📷 Cámara Web":
        img_file_buffer = st.camera_input(
            "Toma una foto para analizar",
            help="Haz click en el botón de la cámara para capturar una imagen"
        )
        image_source = "Cámara"
    else:
        img_file_buffer = st.file_uploader(
            "Sube una imagen para analizar",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Selecciona una imagen desde tu dispositivo"
        )
        image_source = "Archivo"
    
    if img_file_buffer is not None and model is not None:
        # Mostrar imagen original
        img = Image.open(img_file_buffer)
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("**Imagen Original**")
            st.image(img, use_column_width=True)
        
        # Preprocesar imagen
        with st.spinner("🔄 Procesando imagen..."):
            processed_img = preprocess_image(img)
            
            if processed_img is not None:
                # Mostrar imagen procesada
                with col_img2:
                    st.markdown("**Imagen Procesada**")
                    st.image(processed_img, use_column_width=True, clamp=True)
                
                # Realizar predicción
                with st.spinner("🧠 Analizando con IA..."):
                    time.sleep(1)  # Simular procesamiento
                    predictions = predict_image(model, processed_img)
                    
                    if predictions is not None:
                        # Mostrar resultados
                        st.markdown("Resultados del Análisis")
                        detected_class, confidence = display_results(predictions, confianza_minima)
                        
                        # Guardar en historial
                        st.session_state.history.append({
                            'timestamp': datetime.now(),
                            'class': detected_class,
                            'confidence': confidence,
                            'source': image_source
                        })

with col2:
    st.markdown("Historial de Análisis")
    
    if st.session_state.history:
        # Mostrar últimos 5 análisis
        recent_history = st.session_state.history[-5:]
        
        for i, analysis in enumerate(reversed(recent_history)):
            with st.container():
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 0.8rem; border-radius: 8px; margin: 0.3rem 0;">
                    <div style="font-size: 0.8rem; color: #94a3b8;">{analysis['timestamp'].strftime('%H:%M:%S')}</div>
                    <div style="font-weight: 600;">{CLASSES[analysis['class']]}</div>
                    <div style="color: #3b82f6; font-size: 0.9rem;">{analysis['confidence']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Estadísticas del historial
        if len(st.session_state.history) > 1:
            st.markdown("Estadísticas")
            total_analyses = len(st.session_state.history)
            avg_confidence = np.mean([h['confidence'] for h in st.session_state.history])
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Total de Análisis", total_analyses)
            with col_stat2:
                st.metric("Confianza Promedio", f"{avg_confidence:.1%}")
    else:
        st.info("El historial aparecerá aquí después de realizar análisis.")

# Información adicional
with st.expander("ℹ️ Acerca de esta aplicación", expanded=False):
    st.markdown("""
    **Tecnologías Utilizadas:**
    
    - **TensorFlow/Keras**: Framework de deep learning
    - **OpenCV**: Procesamiento de imágenes
    - **Streamlit**: Interfaz web interactiva
    - **Teachable Machine**: Entrenamiento del modelo
    
    **Características:**
    
    - Reconocimiento en tiempo real
    - Múltiples modos de entrada (cámara/archivo)
    - Análisis detallado de confianza
    - Historial de predicciones
    - Visualización interactiva de resultados
    
    **Interpretación de Confianza:**
    
    - 🟢 > 80%: Alta confianza
    - 🟠 60-80%: Confianza media
    - 🔴 < 60%: Baja confianza
    """)
