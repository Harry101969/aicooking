import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
MODEL_NAME = "umass/llama-2-7b-syntod-cooking-assistance"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # Change to torch.float16 if you have a GPU
    device_map="auto"
)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.set_page_config(page_title="Cooking Assistant", layout="centered")
st.title("👩‍🍳 AI Cooking Assistant 🍽️")
st.write("Ask me anything about cooking, recipes, and ingredients!")

user_input = st.text_input("What do you want to cook or ask about?", "How do I make a perfect omelette?")

if st.button("Get Recipe / Advice"):
    with st.spinner("Thinking..."):
        response = generate_response(user_input)
    st.subheader("🍲 Here's what I found:")
    st.write(response)

st.write("💡 Try asking about ingredient substitutes, cooking techniques, or recipes!")


# import streamlit as st
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# import torch

# # Model details
# MODEL_NAME = "umass/llama-2-7b-syntod-cooking-assistance"

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float32,  # Change to torch.float16 if you have a GPU
#     device_map="auto"
# )

# # Initialize pipeline
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# # Streamlit UI
# st.title("🍳 AI Cooking Assistant")
# st.write("Ask me for cooking tips, recipes, or ingredient substitutes!")

# # User input
# user_input = st.text_input("What would you like to cook today?")

# if user_input:
#     st.write("👩‍🍳 Generating recipe...")
#     response = generator(user_input, max_length=150, num_return_sequences=1)
#     st.write("🍽️ Here’s your recipe:")
#     st.write(response[0]['generated_text'])
