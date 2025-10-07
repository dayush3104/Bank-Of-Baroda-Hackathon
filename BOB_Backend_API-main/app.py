import os
import base64
from io import BytesIO
from PIL import Image

import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Initialize Flask App
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow your frontend to communicate with this backend
CORS(app)

# Configure the Gemini API client
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    # Using gemini-1.5-flash as it is fast and cost-effective for this task
    model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20') 
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None

# --- API Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Receives an image and a challenge text, analyzes it with Gemini,
    and returns the result.
    """
    if not model:
        return jsonify({"error": "Gemini model is not configured"}), 500

    # Ensure the request has the correct content type
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    data = request.get_json()
    base64_frame = data.get('image_data')
    challenge_text = data.get('challenge_text')

    # Validate input
    if not base64_frame or not challenge_text:
        return jsonify({"error": "Missing 'image_data' or 'challenge_text' in request"}), 400

    # --- Gemini API Call Logic ---
    try:
        # The prompt remains the same as your original one
        prompt = f'Analyze this image. The user was given the instruction: "{challenge_text}". Determine if the person in the image is a real person correctly performing this action. Respond with only "Yes" or "No".'
        
        # Decode the Base64 string to bytes and open it as an image
        image_bytes = base64.b64decode(base64_frame)
        img = Image.open(BytesIO(image_bytes))

        # Send the prompt and image to the Gemini model
        response = model.generate_content([prompt, img])

        # Return the clean text result
        return jsonify({"result": response.text.strip()})

    except Exception as e:
        import traceback
        # This will print the full, detailed error to your terminal
        print(traceback.format_exc())
        # This will send the specific error message back to the browser
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
        return jsonify({"error": "Failed to analyze image due to an internal error."}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Runs the server on http://127.0.0.1:8000
    app.run(debug=True, port=5000)