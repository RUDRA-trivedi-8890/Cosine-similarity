import requests
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import io
from datetime import datetime, timedelta
from skimage.transform import resize
from sklearn.metrics.pairwise import cosine_similarity
import logging

# --- Boilerplate Flask App Setup ---
app = Flask(__name__)
# Configure logging for better error tracking in production
logging.basicConfig(level=logging.INFO)


# --- Core Logic Function (Unchanged) ---
def get_nasa_similarity_for_location(lat, lng, date_str, threshold=0.75):
    """
    Fetches NASA satellite images for a given location and two dates,
    and calculates the cosine similarity between them.
    """
    nasa_base_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    
    try:
        date1_obj = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        logging.error(f"Invalid date format for: {date_str}")
        return None, "Invalid date format. Please use YYYY-MM-DD."

    date2_obj_week = date1_obj - timedelta(days=7)
    date2_str_week = date2_obj_week.strftime('%Y-%m-%d')

    delta = 0.5
    bbox_str = f"{lng - delta},{lat - delta},{lng + delta},{lat + delta}"
    
    base_params = {
        'SERVICE': 'WMS',
        'REQUEST': 'GetMap',
        'VERSION': '1.3.0',
        'LAYERS': 'MODIS_Terra_CorrectedReflectance_TrueColor',
        'FORMAT': 'image/png',
        'CRS': 'EPSG:4326',
        'BBOX': bbox_str,
        'WIDTH': '1024',
        'HEIGHT': '1024'
    }

    try:
        params1 = base_params.copy()
        params1['TIME'] = date_str
        response1 = requests.get(nasa_base_url, params=params1)
        response1.raise_for_status()
        image1_buffer = response1.content

        params2_week = base_params.copy()
        params2_week['TIME'] = date2_str_week
        response2_week = requests.get(nasa_base_url, params=params2_week)
        response2_week.raise_for_status()
        image2_buffer_week = response2_week.content

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch initial images from NASA GIBS: {e}")
        return None, f"Could not retrieve images from NASA. Error: {e}"

    img1 = plt.imread(io.BytesIO(image1_buffer))
    img2_week = plt.imread(io.BytesIO(image2_buffer_week))
    
    img1_resized = resize(img1, (100, 100), anti_aliasing=True)
    img2_resized_week = resize(img2_week, (100, 100), anti_aliasing=True)

    vec1 = img1_resized.flatten().reshape(1, -1)
    vec2_week = img2_resized_week.flatten().reshape(1, -1)

    similarity_week = cosine_similarity(vec1, vec2_week)
    score = float(similarity_week[0][0])

    if score < threshold:
        logging.info(f"Week 1 score ({score:.4f}) is below threshold ({threshold}). Checking 2 weeks prior.")
        date2_obj_two_weeks = date1_obj - timedelta(days=14)
        date2_str_two_weeks = date2_obj_two_weeks.strftime('%Y-%m-%d')

        try:
            params2_two_weeks = base_params.copy()
            params2_two_weeks['TIME'] = date2_str_two_weeks
            response2_two_weeks = requests.get(nasa_base_url, params=params2_two_weeks)
            response2_two_weeks.raise_for_status()
            image2_buffer_two_weeks = response2_two_weeks.content

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch 2-week fallback image: {e}")
            return score, None

        img2_two_weeks = plt.imread(io.BytesIO(image2_buffer_two_weeks))
        img2_resized_two_weeks = resize(img2_two_weeks, (100, 100), anti_aliasing=True)
        vec2_two_weeks = img2_resized_two_weeks.flatten().reshape(1, -1)
        
        similarity_two_weeks = cosine_similarity(vec1, vec2_two_weeks)
        score = float(similarity_two_weeks[0][0]) 

    return score, None


# --- API Endpoint Definition (MODIFIED) ---
@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity_api():
    """
    API endpoint that interprets the similarity score and provides a validation message.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request. Must be JSON."}), 400

    required_fields = ['lat', 'lng', 'date_str']
    if not all(field in data for field in required_fields):
        return jsonify({"error": f"Missing one or more required fields: {required_fields}"}), 400

    try:
        lat = float(data['lat'])
        lng = float(data['lng'])
        date_str = data['date_str']
        threshold = float(data.get('threshold', 0.75))
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid data type for parameters. {e}"}), 400

    # Call the core logic function
    score, error_message = get_nasa_similarity_for_location(lat, lng, date_str, threshold)

    if error_message:
        return jsonify({"error": error_message}), 502
    
    if score is not None:
        # ✅ NEW: Interpret the score and create a detailed response
        status = ""
        message = ""

        # A score of 1.0 means the images are identical.
        if score == 1.0:
            status = "IDENTICAL"
            message = "No change detected; the areas appear identical."
        # If the score is below the threshold, a significant change is flagged.
        elif score < threshold:
            status = "SIGNIFICANT_CHANGE_DETECTED"
            message = f"Potential deforestation, dumping, or significant land use change detected. The similarity score ({score:.4f}) is below the threshold ({threshold})."
        # Otherwise, the images are similar enough to pass.
        else:
            status = "NO_SIGNIFICANT_CHANGE"
            message = f"No significant change detected. The similarity score ({score:.4f}) is above the threshold ({threshold})."

        # Create the new response object
        response_data = {
            "status": status,
            "message": message,
            "similarity_score": score,
            "threshold_used": threshold
        }
        return jsonify(response_data)
        
    else:
        return jsonify({"error": "An unknown error occurred during processing."}), 500


# --- Main execution block to run the app ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)