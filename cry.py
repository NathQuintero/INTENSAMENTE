import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Configuración de página
st.set_page_config(
    page_title="Detector de sentimientos Intensamente",
    layout="centered",
    page_icon="🧠"
)

# Estilos
st.markdown("""
<style>
body {
    background-color: #F0F8FF;
    font-family: 'Segoe UI', sans-serif;
}
.header-title {
    color: #1F618D;
    font-size: 36px;
    text-align: center;
    font-weight: bold;
}
.author {
    text-align: center;
    font-size: 18px;
    color: #2E4053;
    margin-bottom: 20px;
}
.subtext {
    color: #566573;
    font-size: 20px;
    text-align: center;
    margin-bottom: 30px;
}
.text-box {
    background-color: #ffffff;
    border: 1px solid #D5D8DC;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.result-card {
    background-color: #D6EAF8;
    border: 2px solid #5DADE2;
    border-radius: 10px;
    padding: 25px;
    text-align: center;
    margin-top: 30px;
    color: #154360;
}
.alert {
    background-color: #FADBD8;
    border-left: 6px solid #E74C3C;
    color: #922B21;
    padding: 20px;
    margin-top: 20px;
    border-radius: 8px;
}
.good-news {
    background-color: #D4EFDF;
    border-left: 6px solid #27AE60;
    color: #1E8449;
    padding: 20px;
    margin-top: 20px;
    border-radius: 8px;
}
hr {
    border: none;
    height: 1px;
    background-color: #D5D8DC;
    margin-top: 40px;
}
footer {
    text-align: center;
    font-size: 15px;
    color: #888;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# Título y autor
st.markdown("<h1 class='header-title'>🧠 Detector de sentimientos Intensamente</h1>", unsafe_allow_html=True)
st.markdown("<p class='author'>Hecho por Nathalia Quintero</p>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>¿Cómo estuvo tu día hoy?</p>", unsafe_allow_html=True)

# Cargar modelo
model_path = "AngellyCris/modelo_sentimientos"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Diccionario emociones
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

emotion_translation = {
    "neutral": "neutral",
    "suicidal": "suicida",
    "depressed": "deprimido/a",
    "lonely": "solo/a",
    "disappointment": "decepcionado/a",
    "disgust": "asqueado/a",
    "fear": "asustado/a",
    "anger": "enojado/a",
    "sadness": "triste",
    "hopeless": "sin esperanza",
    "embarrassment": "avergonzado/a",
    "remorse": "arrepentido/a",
    "nervousness": "nervioso/a",
    "grief": "afligido/a",
    "desconocido": "desconocido/a"
}

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

negative_emotions = {
    "suicidal", "depressed", "lonely", "disappointment", "disgust",
    "fear", "anger", "sadness", "hopeless", "embarrassment", "remorse",
    "nervousness", "grief"
}

# Tabs
tab1, tab2 = st.tabs(["💬 Evaluar mi estado", "ℹ️ ¿Cómo funciona esto?"])

with tab1:
    st.markdown("<div class='text-box'>", unsafe_allow_html=True)
    texto_entrada = st.text_area("✏️ Cuéntanos cómo te sientes hoy:", height=150, placeholder="Ej: Me siento cansado pero orgulloso de lo que logré.")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Analizar"):
        if texto_entrada.strip():
            inputs = tokenizer(texto_entrada, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                logits = model(**inputs).logits
            prediccion = torch.argmax(logits, dim=-1).item()
            emocion_ingles = id2emotion.get(prediccion, "desconocido")
            emocion_espanol = emotion_translation.get(emocion_ingles, "desconocido")
            emoji = emotion_emojis.get(emocion_ingles, "❓")

            st.markdown(f"""
            <div class='result-card'>
                <h2>{emoji} Emoción detectada:</h2>
                <h1><strong>{emocion_espanol.capitalize()}</strong></h1>
            </div>
            """, unsafe_allow_html=True)

            if emocion_ingles in negative_emotions:
                st.markdown(f"""
                <div class='alert'>
                    <h4>🚨 Alerta emocional</h4>
                    <p>Detectamos una emoción negativa. No estás solo/a. Respira profundo y recuerda que siempre hay alguien dispuesto a ayudarte.</p>
                    <p><strong>📞 Líneas de atención:</strong></p>
                    <ul>
                        <li>Línea Nacional Colombia: 192 opción 4</li>
                        <li>Línea de la Vida: 01 8000 113 113</li>
                        <li>Habla con alguien de confianza</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='good-news'>
                    <h4>🌈 ¡Nos alegra saber eso!</h4>
                    <p>Tu emoción refleja bienestar. Esperamos que tu día siga lleno de buenas energías 💖</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Por favor, escribe cómo te sientes.")

with tab2:
    st.markdown("""
    ### 🤖 ¿Cómo funciona esto?
    Esta aplicación usa inteligencia artificial para analizar el sentimiento de tu texto.  
    Puedes usarla si quieres saber si tus palabras reflejan emociones como tristeza, ansiedad, enojo o simplemente un estado neutral.

    **Pasos para usarla:**
    1. ✍️ Escribe cómo te sientes.
    2. 🚀 Haz clic en **Analizar**.
    3. 💡 Recibe el resultado con una etiqueta, emoji y sugerencias útiles.

    ---
    Nathalia Quintero, estudiante de Ingeniería de Sistemas de la UNAB.

    > **Nota:** Esta herramienta no reemplaza a un profesional. Si necesitas ayuda, no dudes en buscar apoyo emocional.
    """)

st.markdown("<hr><footer>© Todos los derechos reservados</footer>", unsafe_allow_html=True)
