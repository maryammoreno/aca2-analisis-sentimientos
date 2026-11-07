
import streamlit as st
import joblib
import sqlite3
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# =====================================================

st.set_page_config(
    page_title="Análisis de sentimientos",
    page_icon="⭐",
    layout="centered"
)

# =====================================================
# CARGAR LOS MODELOS
# =====================================================

@st.cache_resource
def cargar_modelos():

    modelo_nb = joblib.load('modelo_sentimientos.pkl')
    vectorizador = joblib.load('vectorizador.pkl')

    modelo_lstm = load_model('modelo_lstm.keras')

    with open('tokenizer_lstm.pkl', 'rb') as archivo:
        tokenizer_lstm = pickle.load(archivo)

    return modelo_nb, vectorizador, modelo_lstm, tokenizer_lstm


modelo_nb, vectorizador, modelo_lstm, tokenizer_lstm = cargar_modelos()

categorias_lstm = {
    0: "Negativa",
    1: "Neutral",
    2: "Positiva"
}

longitud_maxima = 100



# =====================================================
# CONEXIÓN A LA BASE DE DATOS SQLITE
# =====================================================

def consultar_sqlite(consulta):

    with sqlite3.connect("resenas_streamlit.db") as conexion:
        datos = pd.read_sql_query(consulta, conexion)

    return datos


# =====================================================
# FUNCIONES PARA REALIZAR PREDICCIONES
# =====================================================

def predecir_naive_bayes(texto):

    texto_limpio = " ".join(texto.lower().split())

    texto_vectorizado = vectorizador.transform([texto_limpio])

    sentimiento = modelo_nb.predict(texto_vectorizado)[0]

    probabilidades = modelo_nb.predict_proba(texto_vectorizado)[0]

    resultados = dict(zip(modelo_nb.classes_, probabilidades))

    return sentimiento, resultados


def predecir_lstm(texto):

    secuencias = tokenizer_lstm.texts_to_sequences([texto])

    secuencias_ajustadas = pad_sequences(
        secuencias,
        maxlen=longitud_maxima,
        padding='post',
        truncating='post'
    )

    probabilidades = modelo_lstm.predict(
        secuencias_ajustadas,
        verbose=0
    )[0]

    indice = np.argmax(probabilidades)

    sentimiento = categorias_lstm[indice]

    resultados = {
        categorias_lstm[i]: float(probabilidades[i])
        for i in range(3)
    }

    return sentimiento, resultados


def mostrar_resultado(nombre_modelo, sentimiento, probabilidades):

    st.write(f"### Resultado de {nombre_modelo}")

    if sentimiento == "Positiva":
        st.success(f"Sentimiento identificado: {sentimiento} 😊")

    elif sentimiento == "Neutral":
        st.warning(f"Sentimiento identificado: {sentimiento} 😐")

    else:
        st.error(f"Sentimiento identificado: {sentimiento} 😞")

    st.write("**Probabilidades estimadas por el modelo:**")

    for categoria in ["Negativa", "Neutral", "Positiva"]:

        probabilidad = probabilidades[categoria]

        st.write(f"{categoria}: {probabilidad * 100:.2f}%")

        st.progress(float(probabilidad))


# =====================================================
# TÍTULO Y DESCRIPCIÓN
# =====================================================

st.title("⭐ Análisis de sentimientos de productos Amazon")

st.write(
    "Esta aplicación utiliza dos modelos de aprendizaje automático, "
    "Naive Bayes y una red neuronal LSTM, para clasificar las reseñas "
    "de los consumidores en positivas, neutrales o negativas."
)

st.info("Las reseñas deben escribirse en inglés.")


# =====================================================
# CREAR LAS PESTAÑAS
# =====================================================

tab1, tab2, tab3 = st.tabs([
    "📝 Analizar reseña",
    "📊 Resultados de los modelos",
    "🗄️ Base de datos"
])


# =====================================================
# PESTAÑA 1: ANALIZAR RESEÑAS
# =====================================================

with tab1:

    st.subheader("Clasificación de sentimientos")

    st.write(
        "Escribe una reseña sobre un producto Amazon y selecciona "
        "el modelo que deseas utilizar para analizar su sentimiento."
    )

    resena = st.text_area(
        "Escribe una reseña sobre un producto:",
        placeholder="Example: This tablet is amazing and easy to use."
    )

    modelo_seleccionado = st.selectbox(
        "Selecciona el modelo:",
        [
            "Naive Bayes",
            "LSTM",
            "Comparar ambos modelos"
        ]
    )

    if st.button("Analizar sentimiento"):

        if resena.strip():

            if modelo_seleccionado == "Naive Bayes":

                sentimiento, probabilidades = predecir_naive_bayes(resena)

                mostrar_resultado(
                    "Naive Bayes",
                    sentimiento,
                    probabilidades
                )

            elif modelo_seleccionado == "LSTM":

                sentimiento, probabilidades = predecir_lstm(resena)

                mostrar_resultado(
                    "LSTM",
                    sentimiento,
                    probabilidades
                )

            else:

                sentimiento_nb, probabilidades_nb = predecir_naive_bayes(resena)

                sentimiento_lstm, probabilidades_lstm = predecir_lstm(resena)

                mostrar_resultado(
                    "Naive Bayes",
                    sentimiento_nb,
                    probabilidades_nb
                )

                st.divider()

                mostrar_resultado(
                    "LSTM",
                    sentimiento_lstm,
                    probabilidades_lstm
                )

                st.divider()

                st.subheader("Comparación de predicciones")

                st.write(f"**Naive Bayes:** {sentimiento_nb}")
                st.write(f"**LSTM:** {sentimiento_lstm}")

                if sentimiento_nb == sentimiento_lstm:

                    st.info(
                        "Los dos modelos identificaron el mismo sentimiento."
                    )

                else:

                    st.warning(
                        "Los modelos identificaron sentimientos diferentes."
                    )

        else:

            st.warning(
                "Por favor, escribe una reseña antes de analizar."
            )


# =====================================================
# PESTAÑA 2: RESULTADOS DE LOS MODELOS
# =====================================================

with tab2:

    st.subheader("Evaluación de los modelos")

    st.write(
        "En esta sección se comparan los resultados obtenidos "
        "durante la evaluación de Naive Bayes y la red neuronal LSTM."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Exactitud Naive Bayes", "93.25 %")

    with col2:
        st.metric("Exactitud LSTM", "84.36 %")

    st.write(
        "Ambos modelos fueron evaluados utilizando 6.923 reseñas "
        "correspondientes al conjunto de prueba."
    )

    # -------------------------------------------------
    # COMPARACIÓN DEL F1-SCORE
    # -------------------------------------------------

    st.subheader("Comparación del F1-score por categoría")

    categorias = ["Negativa", "Neutral", "Positiva"]

    f1_nb = [28.2, 9.9, 96.6]
    f1_lstm = [37.2, 18.6, 91.9]

    x = np.arange(len(categorias))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    barras_nb = ax.bar(
        x - ancho / 2,
        f1_nb,
        ancho,
        label="Naive Bayes"
    )

    barras_lstm = ax.bar(
        x + ancho / 2,
        f1_lstm,
        ancho,
        label="LSTM"
    )

    ax.set_title("Comparación del F1-score por categoría")
    ax.set_ylabel("F1-score (%)")
    ax.set_xlabel("Categorías de sentimientos")

    ax.set_xticks(x)
    ax.set_xticklabels(categorias)

    ax.set_ylim(0, 110)
    ax.legend()

    for barras in [barras_nb, barras_lstm]:

        for barra in barras:

            altura = barra.get_height()

            ax.annotate(
                f"{altura:.1f}%",
                xy=(
                    barra.get_x() + barra.get_width() / 2,
                    altura
                ),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8
            )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write(
        "Naive Bayes obtuvo mejores resultados en las opiniones positivas, "
        "mientras que LSTM presentó mejores resultados en las categorías "
        "negativa y neutral."
    )

    # -------------------------------------------------
    # MATRICES DE CONFUSIÓN
    # -------------------------------------------------

    st.subheader("Matrices de confusión")

    modelo_grafica = st.selectbox(
        "Selecciona el modelo para visualizar su matriz:",
        ["Naive Bayes", "LSTM"]
    )

    if modelo_grafica == "Naive Bayes":

        matriz = np.array([
            [31, 14, 117],
            [6, 18, 276],
            [21, 33, 6407]
        ])

    else:

        matriz = np.array([
            [94, 37, 31],
            [53, 98, 149],
            [196, 617, 5648]
        ])

    fig2, ax2 = plt.subplots(figsize=(7, 5))

    imagen = ax2.imshow(
        matriz,
        cmap="Blues"
    )

    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))

    ax2.set_xticklabels(categorias)
    ax2.set_yticklabels(categorias)

    ax2.set_xlabel("Sentimiento predicho")
    ax2.set_ylabel("Sentimiento real")

    ax2.set_title(
        f"Matriz de confusión del modelo {modelo_grafica}"
    )

    for i in range(3):

        for j in range(3):

            ax2.text(
                j,
                i,
                matriz[i, j],
                ha="center",
                va="center",
                color="white" if matriz[i, j] > matriz.max()/2 else "black"
            )

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # -------------------------------------------------
    # CONCLUSIONES
    # -------------------------------------------------

    st.subheader("Conclusión de la evaluación")

    st.write(
        "Naive Bayes alcanzó una exactitud general del 93,25 %, "
        "mientras que LSTM obtuvo un 84,36 %. Sin embargo, "
        "la red neuronal presentó mejores resultados en la "
        "identificación de las categorías negativa y neutral."
    )

    st.write(
        "Estos resultados pueden estar relacionados con el "
        "desequilibrio del conjunto de datos, donde predominan "
        "las reseñas positivas. Aunque ambos modelos permiten "
        "clasificar sentimientos, todavía presentan dificultades "
        "para identificar correctamente algunas opiniones "
        "negativas y neutrales."
    )


# =====================================================
# PESTAÑA 3: BASE DE DATOS SQLITE
# =====================================================

with tab3:

    st.subheader("Reseñas almacenadas en SQLite")

    st.write(
        "En esta sección se presenta la información de las "
        "reseñas de productos Amazon almacenadas en la base "
        "de datos SQLite, después del proceso de limpieza."
    )

    # Consultar cantidad total de reseñas
    total = consultar_sqlite(
        "SELECT COUNT(*) AS total FROM resenas_limpias"
    )

    st.metric(
        "Total de reseñas almacenadas",
        f"{int(total['total'].iloc[0]):,}"
    )

    # Consultar distribución de sentimientos
    distribucion = consultar_sqlite(
        """
        SELECT sentimiento, COUNT(*) AS cantidad
        FROM resenas_limpias
        GROUP BY sentimiento
        """
    )

    st.subheader("Distribución de sentimientos")

    st.dataframe(distribucion, hide_index=True)

    # Gráfica de distribución
    fig3, ax3 = plt.subplots(figsize=(8, 4))

    ax3.bar(
        distribucion["sentimiento"],
        distribucion["cantidad"]
    )

    ax3.set_title("Cantidad de reseñas por sentimiento")
    ax3.set_xlabel("Sentimiento")
    ax3.set_ylabel("Cantidad de reseñas")

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.write(
        "Se observa que la mayoría de las reseñas almacenadas "
        "corresponden a opiniones positivas, mientras que las "
        "categorías negativa y neutral tienen menos registros."
    )

    # Consultar una muestra de las reseñas
    st.subheader("Muestra de reseñas almacenadas")

    muestra = consultar_sqlite(
        """
        SELECT sentimiento, texto_limpio
        FROM resenas_limpias
        LIMIT 10
        """
    )

    st.dataframe(
        muestra,
        hide_index=True,
        use_container_width=True
    )
