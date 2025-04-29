import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Cargar el modelo y el tokenizer
model_path = 'pysentimiento/robertuito-base-uncased-save'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Lista de emociones (ordenadas como tu dataset)
emotions = [
    "neutral", "suicidal", "depressed", "lonely", "disappointment",
    "disgust", "fear", "anger", "sadness", "hopeless",
    "embarrassment", "remorse", "nervousness", "grief"
]

# Página web
st.title("🧠 Detector de Emociones de Pacientes")
st.write("Escribe un texto y te diremos qué emoción predomina.")

# Caja de texto
text_input = st.text_area("✍️ Escribe aquí tu texto:", height=150)

# Botón para predecir
if st.button("Predecir emoción"):
    if text_input.strip() == "":
        st.warning("Por favor escribe algo.")
    else:
        # Tokenizar entrada
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
        
        # Hacer predicción
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)

        # Obtener predicción
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_emotion = emotions[pred_idx]
        confidence = probs[0, pred_idx].item()

        # Mostrar resultados
        st.success(f"🎯 **Emoción detectada:** {pred_emotion}")
        st.info(f"🔎 **Confianza:** {confidence*100:.2f}%")
