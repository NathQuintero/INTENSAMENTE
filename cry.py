import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import safetensors

# 🎯 Configuración de la página
st.set_page_config(
    page_title="Evaluador PPE Inteligente",
    layout="wide",
    page_icon="🧠"
)

# 🌟 Estilos CSS para mejorar la estética
st.markdown("""
<style>
body {
    background-color: #F4F6F6;
}
.header-title {
    font-family: 'Helvetica', sans-serif;
    color: #2E86C1;
    font-size: 36px;
    text-align: center;
    margin-top: 0;
}
.subtext {
    font-family: 'Helvetica', sans-serif;
    color: #566573;
    font-size: 18px;
    text-align: center;
}
.text-box {
    background-color: #FBFCFC;
    border-radius: 10px;
    padding: 20px;
}
.result-card {
    background-color: #E8F8F5;
    border: 2px solid #1ABC9C;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ✨ Encabezado superior
st.markdown("<h1 class='header-title'>🧠 Evaluador Inteligente de Sentimientos</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Todos los derechos reservados ©️ | Análisis de emociones a partir de texto</p>", unsafe_allow_html=True)

# 📦 Cargar el modelo
model_path = "AngellyCris/modelo_sentimientos"
model = AutoModelForSequenceClassification.from_pretrained(model_path, use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 📖 Instrucciones
with st.expander("📚 FUNCIONAMIENTO"):
    st.markdown("""
    - ✍️ Escribe un **texto breve** describiendo cómo te sientes.
    - 🔎 Presiona **Analizar** y descubre el sentimiento predominante.
    """)

# 📄 Cuadro de texto para ingresar la descripción
st.markdown("<div class='text-box'>", unsafe_allow_html=True)
texto_entrada = st.text_area("✏️ ESCRIBE AQUÍ COMO TE SIENTES:", height=200, placeholder="Me siento feliz de estar aquí...")
st.markdown("</div>", unsafe_allow_html=True)

# Diccionario de emociones
id2emotion = {
    0: "neutral",
    1: "suicidal",
    2: "depressed",
    3: "lonely",
    4: "disappointment",
    5: "disgust",
    6: "fear",
    7: "anger",
    8: "sadness",
    9: "hopeless",
    10: "embarrassment",
    11: "remorse",
    12: "nervousness",
    13: "grief"
}

# Emojis para emociones
emotion_emojis = {
    "neutral": "😐",
    "suicidal": "🆘",
    "depressed": "😞",
    "lonely": "🥺",
    "disappointment": "😔",
    "disgust": "🤢",
    "fear": "😨",
    "anger": "😠",
    "sadness": "😢",
    "hopeless": "😩",
    "embarrassment": "😳",
    "remorse": "😥",
    "nervousness": "😬",
    "grief": "😭",
    "desconocido": "❓"
}

# 🔍 Botón para analizar el texto
if st.button("🚀 Analizar Texto"):
    if texto_entrada.strip():
        # Tokenización y predicción
        inputs = tokenizer(texto_entrada, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            logits = model(**inputs).logits
        prediccion = torch.argmax(logits, dim=-1).item()

        # Traducir predicción a emoción
        emocion_predicha = id2emotion.get(prediccion, "desconocido")
        emoji = emotion_emojis.get(emocion_predicha, "❓")

        # 🎨 Mostrar resultado
        st.markdown(f"""
        <div class='result-card'>
            <h2>{emoji} Emoción detectada:</h2>
            <h1><strong>{emocion_predicha.capitalize()}</strong></h1>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Por favor, escribe cómo te sientes.")

# 🎨 Pie de página decorativo
st.markdown("<hr><center>Creado con ❤️ por Mí , soporta check</center>", unsafe_allow_html=True)
