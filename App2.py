import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
MODEL_NAME = "umass/llama-2-7b-syntod-cooking-assistance"

@st.cache_resource  # Caches the model to avoid reloading on rerun
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  
        device_map="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

pipe = load_model()

# Function to generate response
def generate_response(prompt):
    response = pipe(prompt, max_length=150, num_return_sequences=1, do_sample=True)
    return response[0]["generated_text"]

# Streamlit UI
st.set_page_config(page_title="AI Cooking Assistant", layout="centered")
st.title("👩‍🍳 AI Cooking Assistant 🍽️")
st.write("Ask me anything about cooking, recipes, and ingredients!")

user_input = st.text_input("What do you want to cook or ask about?", "How do I make a perfect omelette?")

if st.button("Get Recipe / Advice"):
    with st.spinner("Cooking up a response..."):
        response = generate_response(user_input)
    st.subheader("🍲 Here's what I found:")
    st.write(response)

st.write("💡 Try asking about ingredient substitutes, cooking techniques, or recipes!")
