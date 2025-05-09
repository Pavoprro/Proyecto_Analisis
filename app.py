# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Recomendación Netflix",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("🎬 Recomendador Personalizado de Netflix")
st.markdown("""
Descubre nuevas películas y series basadas en tus preferencias.
¡El sistema aprende de tus gustos para sugerirte contenido similar!
""")

# Carga de recursos con caché
@st.cache_resource
def load_resources():
    """Carga modelos y encoders"""
    try:
        resources = {
            'model': joblib.load('netflix_model.joblib'),
            'encoder': joblib.load('genre_encoder.joblib'),
            'metadata': joblib.load('model_metadata.joblib')
        }
        st.success("✅ Modelos cargados correctamente")
        return resources
    except Exception as e:
        st.error(f"❌ Error cargando modelos: {str(e)}")
        st.stop()

@st.cache_data
def load_data():
    """Carga y preprocesa datos básicos"""
    try:
        df = pd.read_csv('netflix_titles.csv', usecols=[
            'title', 'type', 'duration', 'listed_in', 'release_year', 'rating'
        ])
        
        # Conversión de duración
        df['duration'] = pd.to_numeric(df['duration'].str.extract('(\d+)')[0])
        df['duration'] = df['duration'].fillna(df['duration'].median())
        
        return df
    except Exception as e:
        st.error(f"❌ Error cargando datos: {str(e)}")
        st.stop()

# Carga inicial
resources = load_resources()
df = load_data()

# Sidebar con información
with st.sidebar:
    st.header("ℹ️ Información del Sistema")
    st.markdown("""
    **Modelo actual:** Árbol de Decisión  
    **Características usadas:**
    - Año de lanzamiento
    - Duración
    - Género principal
    
    **Precisión estimada:** 82-85%
    """)
    
    st.divider()
    st.write("🔍 Datos técnicos:")
    st.json({
        "features": resources['metadata']['feature_names'],
        "model_type": resources['metadata']['model_type'],
        "genres_encoded": len(resources['encoder'].classes_)
    })

# Función para procesar recomendaciones
def get_recommendations(favorite_titles, n_recommendations=5):
    """Genera recomendaciones basadas en títulos favoritos"""
    try:
        # Paso 1: Procesar títulos favoritos
        features = []
        found_titles = []
        
        for title in favorite_titles:
            match = df[df['title'].str.lower() == title.lower()]
            if not match.empty:
                # Codificar género del título encontrado
                genre = match['listed_in'].iloc[0].split(',')[0]
                genre_encoded = resources['encoder'].transform([genre])[0]
                
                features.append({
                    'release_year': match['release_year'].iloc[0],
                    'duration': match['duration'].iloc[0],
                    'genre_encoded': genre_encoded
                })
                found_titles.append(title)
        
        if not features:
            st.warning("⚠️ No se encontraron coincidencias con tus favoritos")
            return None
        
        # Paso 2: Calcular características promedio
        avg_features = pd.DataFrame(features).mean()
        
        # Paso 3: Codificar géneros para todo el dataset
        df['genre_encoded'] = resources['encoder'].transform(
            df['listed_in'].str.split(',').str[0].fillna('Unknown'))
        
        # Paso 4: Calcular predicciones y similitud
        X = df[resources['metadata']['feature_names']]
        df['prediction'] = resources['model'].predict_proba(X)[:, 1]
        
        # Fórmula de similitud mejorada
        df['similarity'] = (
            0.5 * (1 - abs((df['release_year'] - avg_features['release_year']) / 100)) +
            0.3 * (1 - abs((df['duration'] - avg_features['duration']) / 180)) +
            0.2 * (df['genre_encoded'] == round(avg_features['genre_encoded']))
        )
        
        # Paso 5: Combinar métricas y filtrar
        df['score'] = (df['prediction'] * df['similarity']).round(3)
        recommendations = df[
            ~df['title'].str.lower().isin([t.lower() for t in found_titles])
        ].sort_values('score', ascending=False).head(n_recommendations)
        
        return recommendations[['title', 'type', 'duration', 'release_year', 'listed_in', 'score']]
    
    except Exception as e:
        st.error(f"Error generando recomendaciones: {str(e)}")
        return None

# Interfaz principal
with st.form("recommendation_form"):
    st.subheader("📌 Ingresa tus títulos favoritos")
    favorites = st.text_area(
        "Escribe uno o más títulos (separados por línea):",
        height=150,
        placeholder="Ejemplo:\nStranger Things\nLa Casa de Papel\nEl Juego del Calamar"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        n_recs = st.slider(
            "Número de recomendaciones:",
            min_value=1,
            max_value=20,
            value=10,
            help="Selecciona cuántas recomendaciones deseas ver"
        )
    
    with col2:
        st.write("")  # Espacio en blanco para alineación
        submit_button = st.form_submit_button(
            "🚀 Obtener Recomendaciones",
            type="primary"
        )

# Procesar cuando se envía el formulario
if submit_button:
    favorite_titles = [t.strip() for t in favorites.split('\n') if t.strip()]
    
    if not favorite_titles:
        st.warning("Por favor ingresa al menos un título favorito")
    else:
        with st.spinner("Analizando tus preferencias..."):
            recommendations = get_recommendations(favorite_titles, n_recs)
            
            if recommendations is not None and not recommendations.empty:
                # Mostrar resultados en dos formatos
                tab1, tab2 = st.tabs(["📊 Vista Tabular", "🎥 Vista Tarjetas"])
                
                with tab1:
                    st.success(f"✨ Top {len(recommendations)} recomendaciones para ti:")
                    st.dataframe(
                        recommendations.rename(columns={
                            'title': 'Título',
                            'type': 'Tipo',
                            'duration': 'Duración (min)',
                            'release_year': 'Año',
                            'listed_in': 'Géneros',
                            'score': 'Puntuación'
                        }),
                        hide_index=True,
                        column_config={
                            "Puntuación": st.column_config.ProgressColumn(
                                format="%.3f",
                                min_value=0,
                                max_value=1,
                            )
                        },
                        use_container_width=True
                    )
                
                with tab2:
                    st.subheader("🎬 Vista previa de recomendaciones")
                    cols = st.columns(2)
                    
                    for idx, (_, row) in enumerate(recommendations.iterrows()):
                        with cols[idx % 2]:
                            with st.container(border=True):
                                st.markdown(f"""
                                <style>
                                .recommendation-card {{
                                    padding: 15px;
                                    border-radius: 10px;
                                    background-color: #0E1117;
                                    margin-bottom: 15px;
                                }}
                                </style>
                                
                                <div class="recommendation-card">
                                    <h4>{row['title']}</h4>
                                    <p><b>Tipo:</b> {row['type']}</p>
                                    <p><b>Duración:</b> {int(row['duration'])} min</p>
                                    <p><b>Año:</b> {int(row['release_year'])}</p>
                                    <p><b>Género principal:</b> {row['listed_in'].split(',')[0]}</p>
                                    <p><b>Relevancia:</b> {row['score']:.0%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Sección de análisis
                with st.expander("📈 Análisis de tus preferencias"):
                    st.subheader("Tus patrones de preferencia")
                    
                    # Gráfico de características promedio
                    fig, ax = plt.subplots(figsize=(10, 4))
                    features = recommendations[['duration', 'release_year']].mean()
                    pd.Series({
                        'Duración (min)': features['duration'],
                        'Año de lanzamiento': features['release_year']
                    }).plot(kind='bar', ax=ax, color=['#E50914', '#221F1F'])
                    plt.title("Promedio de características recomendadas")
                    st.pyplot(fig)
                    
                    # Distribución de géneros
                    genre_counts = recommendations['listed_in'].str.split(',').explode().str.strip().value_counts().head(5)
                    st.write("**Géneros más frecuentes en tus recomendaciones:**")
                    st.bar_chart(genre_counts)
            
            elif recommendations is not None and recommendations.empty:
                st.warning("No se encontraron recomendaciones adecuadas. Prueba con otros títulos.")
            else:
                st.error("Ocurrió un error al generar recomendaciones. Por favor intenta nuevamente.")

# Footer
st.divider()
st.caption("""
Sistema de recomendación basado en aprendizaje automático. 
Los resultados son estimaciones basadas en patrones de contenido.
""")
