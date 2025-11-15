import os
import google.generativeai as genai

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY"),
    transport="rest"
)

print("GENAI client configured. Transport should be 'rest'.")

try:
    models = genai.list_models()
    print("Available models:")
    for m in models:
        print("-", m.name)
except Exception as e:
    print("List models error:", e)
