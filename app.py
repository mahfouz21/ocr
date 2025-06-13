# app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import pytesseract
from PIL import Image
import tempfile
import os
import uuid
import fitz
from pymongo import MongoClient
import ollama
from datetime import datetime, UTC
import google.generativeai as genai  # Gemini API

app = Flask(__name__)

MONGODB_URI = "mongodb+srv://mahfouz0808:WIRdt0sIOONB5lwD@cluster0.9di0ahd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "prescription_db"
COLLECTION_NAME = "prescriptions"

class ImageProcessor:
    @staticmethod
    def preprocess_image(image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, processed = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        return processed

    @staticmethod
    def extract_text(image_path):
        image = ImageProcessor.preprocess_image(image_path)
        text = pytesseract.image_to_string(image, lang="ara+eng")
        return text

    @staticmethod
    def extract_images_from_pdf(pdf_path):
        doc = fitz.open(pdf_path)
        images = []
        for page_number in range(len(doc)):
            pix = doc[page_number].get_pixmap(dpi=300)
            image_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            images.append(image_array)
        return images

class MongoDBManager:
    def __init__(self, connection_string, db_name, collection_name):
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def save_prescription(self, prescription_data):
        document = {
            "prescription_id": str(uuid.uuid4()),
            "text_content": prescription_data,
            "created_at": datetime.now(UTC),
            "status": "processed"
        }
        result = self.collection.insert_one(document)
        return str(result.inserted_id)

# --- GeminiTextCorrector class ---
class GeminiTextCorrector:
    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def correct_text(self, ocr_text):
        prompt = """
You are a professional medical prescription assistant.
You should always answer the question, no matter the quality of the input text.
Make sure to structure your answer clearly and provide all the requested details!
Don't give answers like: I can't help with that. Is there anything else I can help you with?

You should Answer No Matter what! Even if it is not well!
Please analyze the following text carefully and provide the following details in a structured format:

- Doctor's Name
- Patient's Name
- Date of Prescription
- Medicines:
* Name
* Dosage
* Instructions

Here is the text to analyze:
""" + ocr_text
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini Error: {e}"

# Set your Gemini API key here
GEMINI_API_KEY = "AIzaSyAHPhQErRVwU_c8A-5cjN4UZH9FyCb6Quk"
gemini_corrector = GeminiTextCorrector(api_key=GEMINI_API_KEY, model_name="gemini-1.5-flash")

# class LLaMaTextCorrector:
#     @staticmethod
#     def correct_text(ocr_text):
#         """Use LLaMa to correct and structure OCR text."""
#         chat_history = [
#             {"role": "system", "content": """You are a professional medical prescription assistant.
#             You should always answer the question, no matter the quality of the input text.
#             Make sure to structure your answer clearly and provide all the requested details!
#             Don't give answers like: I can't help with that. Is there anything else I can help you with?"""},
#             {"role": "user", "content": f"""
#             You should Answer No Matter what! Even if it is not well!
#             Please analyze the following text carefully and provide the following details in a structured format:
            
#             - Doctor's Name
#             - Patient's Name
#             - Date of Prescription
#             - Medicines:
#             * Name
#             * Dosage
#             * Instructions
            
#             Here is the text to analyze:
#             {ocr_text}
#             """}
#         ]
#         try:
#             # answer = ollama.chat(model="llama3.2", messages=chat_history, stream=True)
#             # corrected_text = ""
#             # for token_dict in answer:
#             #     token = token_dict["message"]["content"]
#             #     corrected_text += token
#             # return corrected_text
#             pass  # Ollama code commented out for Gemini usage
#         except Exception as e:
#             return f"LLaMa Error: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/OCR', methods=['POST'])
def ocr():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
        file.save(temp.name)
        temp_path = temp.name

    recognized_text = ImageProcessor.extract_text(temp_path)
    structured_text = gemini_corrector.correct_text(recognized_text)
    mongo_manager = MongoDBManager(MONGODB_URI, DB_NAME, COLLECTION_NAME)
    prescription_id = mongo_manager.save_prescription(structured_text)

    os.remove(temp_path)
    return render_template(
        'index.html',
        prescription_id=prescription_id,
        recognized_text=recognized_text,
        structured_text=structured_text
    )

if __name__ == '__main__':
    app.run(debug=True)