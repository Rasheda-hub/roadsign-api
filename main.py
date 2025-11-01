from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import requests
import io
from config import ROBOFLOW_API_URL, ROBOFLOW_API_KEY, REQUEST_TIMEOUT
from utils import normalize_roboflow_response

app = FastAPI(
    title="Road Sign Detection (Roboflow)",
    description="FastAPI wrapper that sends images to a Roboflow traffic sign model and returns cleaned JSON detections.",
    version="1.0"
)

@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Serve the main UI page.
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Road Sign Detection</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e8ba3 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
                max-width: 800px;
                width: 100%;
            }
            
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            
            .upload-area {
                border: 3px dashed #2a5298;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                transition: all 0.3s ease;
                cursor: pointer;
                background: #f0f7ff;
            }
            
            .upload-area:hover {
                border-color: #1e3c72;
                background: #e6f2ff;
            }
            
            .upload-area.dragover {
                border-color: #1e3c72;
                background: #d9edff;
                transform: scale(1.02);
            }
            
            #fileInput {
                display: none;
            }
            
            .upload-icon {
                font-size: 4em;
                margin-bottom: 20px;
            }
            
            .upload-text {
                color: #2a5298;
                font-size: 1.2em;
                font-weight: 600;
                margin-bottom: 10px;
            }
            
            .upload-hint {
                color: #999;
                font-size: 0.9em;
            }
            
            .btn {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 30px;
                font-size: 1.1em;
                font-weight: 600;
                cursor: pointer;
                margin-top: 20px;
                transition: all 0.3s ease;
                display: none;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(42, 82, 152, 0.4);
            }
            
            .btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            #preview {
                margin-top: 30px;
                display: none;
            }
            
            #previewImage {
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            
            #results {
                margin-top: 30px;
                display: none;
            }
            
            .result-card {
                background: #f0f7ff;
                border-left: 4px solid #2a5298;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 15px;
            }
            
            .result-title {
                color: #1e3c72;
                font-weight: 600;
                font-size: 1.2em;
                margin-bottom: 10px;
            }
            
            .detection-item {
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            .detection-class {
                font-weight: 600;
                color: #333;
                font-size: 1.1em;
            }
            
            .detection-confidence {
                color: #2a5298;
                font-weight: 600;
                margin-left: 10px;
            }
            
            .detection-description {
                color: #666;
                font-size: 0.9em;
                margin-top: 8px;
                line-height: 1.5;
                font-style: italic;
            }
            
            .loading {
                text-align: center;
                padding: 20px;
                display: none;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #2a5298;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .error {
                background: #ffe6e6;
                border-left: 4px solid #ff4444;
                color: #cc0000;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚦 Road Sign Detection</h1>
            <p class="subtitle">Upload an image to detect road signs using AI</p>
            
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📸</div>
                <div class="upload-text">Click to upload or drag and drop</div>
                <div class="upload-hint">Supports: JPG, PNG, JPEG</div>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            
            <center>
                <button class="btn" id="detectBtn">Detect Road Signs</button>
            </center>
            
            <div id="preview">
                <h3 style="margin-bottom: 15px; color: #333;">Preview:</h3>
                <img id="previewImage" alt="Preview">
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 15px; color: #2a5298; font-weight: 600;">Analyzing image...</p>
            </div>
            
            <div class="error" id="error"></div>
            
            <div id="results"></div>
        </div>
        
        <script>
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const detectBtn = document.getElementById('detectBtn');
            const preview = document.getElementById('preview');
            const previewImage = document.getElementById('previewImage');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            const errorDiv = document.getElementById('error');
            
            let selectedFile = null;
            
            // Click to upload
            uploadArea.addEventListener('click', () => fileInput.click());
            
            // File selection
            fileInput.addEventListener('change', (e) => {
                handleFile(e.target.files[0]);
            });
            
            // Drag and drop
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                handleFile(e.dataTransfer.files[0]);
            });
            
            function handleFile(file) {
                if (!file || !file.type.startsWith('image/')) {
                    showError('Please select a valid image file');
                    return;
                }
                
                selectedFile = file;
                
                // Show preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewImage.src = e.target.result;
                    preview.style.display = 'block';
                    detectBtn.style.display = 'inline-block';
                    results.style.display = 'none';
                    errorDiv.style.display = 'none';
                };
                reader.readAsDataURL(file);
            }
            
            // Detect button
            detectBtn.addEventListener('click', async () => {
                if (!selectedFile) return;
                
                const formData = new FormData();
                formData.append('file', selectedFile);
                
                loading.style.display = 'block';
                results.style.display = 'none';
                errorDiv.style.display = 'none';
                detectBtn.disabled = true;
                
                try {
                    const response = await fetch('/detect', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        displayResults(data.raw);
                    } else {
                        showError(data.detail || 'Detection failed');
                    }
                } catch (error) {
                    showError('Network error: ' + error.message);
                } finally {
                    loading.style.display = 'none';
                    detectBtn.disabled = false;
                }
            });
            
            function getSignDescription(signClass) {
                const descriptions = {
                    // Stop signs
                    'stop': 'Come to a full stop and check for traffic.',
                    'stop sign': 'Come to a full stop and check for traffic.',
                    'stop_sign': 'Come to a full stop and proceed only when safe.',
                    
                    // Parking
                    'no parking': 'Parking is not allowed in this area.',
                    
                    // Turns
                    'no u-turn': 'U-turns are prohibited here.',
                    'no left turn': 'Left turns are not permitted at this point.',
                    'no right turn': 'Right turns are not permitted at this point.',
                    'turn left': 'You are required to turn left.',
                    'turn right': 'You are required to turn right.',
                    'turn left ahead': 'Left turn ahead — slow down and prepare to turn.',
                    'turn right ahead': 'Right turn ahead — slow down and prepare to turn.',
                    
                    // Directional
                    'one way street': 'Traffic flows only in one direction.',
                    'one way': 'Traffic flows only in one direction.',
                    'straight ahead only': 'You may only go straight — no turns allowed.',
                    'go straight or turn right': 'You may proceed straight or turn right — no left turn allowed.',
                    'go straight or turn left': 'You may proceed straight or turn left — no right turn allowed.',
                    
                    // Yield/Give Way
                    'yield': 'Slow down and let other vehicles go first.',
                    'give way': 'Slow down and give priority to other vehicles before proceeding.',
                    'give way to oncoming': 'Yield to oncoming traffic.',
                    
                    // Pedestrian
                    'pedestrian crossing': 'Slow down and watch for people crossing.',
                    'children crossing': 'Slow down; children may cross the road here.',
                    'beware of children': 'Children may be crossing — slow down and stay alert.',
                    'bicycle crossing': 'Bicycles crossing ahead — slow down and watch.',
                    
                    // Speed limits
                    'speed limit 20 kmph': 'Maximum speed allowed is 20 km/h — drive slowly.',
                    'speed limit 30 kmph': 'Maximum speed allowed is 30 km/h.',
                    'speed limit 30 km/h': 'Maximum speed is 30 km/h in this area.',
                    'speed limit 40 km/h': 'Maximum speed is 40 km/h in this area.',
                    'speed limit 50 kmph': 'Maximum speed allowed is 50 km/h.',
                    'speed limit 50 km/h': 'Maximum speed is 50 km/h in this area.',
                    'speed limit 60 km/h': 'Maximum speed is 60 km/h in this area.',
                    'speed limit 70 km/h': 'Maximum speed is 70 km/h in this area.',
                    'speed limit 80 km/h': 'Maximum speed is 80 km/h in this area.',
                    '50 mph speed limit': 'Maximum speed allowed is 50 mph — do not exceed.',
                    'end of all speed and passing limits': 'Previous speed and passing limits are lifted — drive responsibly.',
                    'end of speed limit': 'Previous speed limit ends — adjust accordingly.',
                    
                    // Road conditions
                    'roadworks ahead': 'Construction work ahead — slow down.',
                    'slippery road': 'Road may be slippery when wet — proceed carefully.',
                    'slippery road ahead': 'Road may be slippery — reduce speed and avoid sudden turns or brakes.',
                    'bumpy road': 'Uneven surface ahead — reduce your speed.',
                    'uneven road': 'Road surface ahead is uneven — reduce speed to avoid bumps.',
                    
                    // Entry restrictions
                    'no entry': 'Do not enter — road is closed or one-way against you.',
                    'no_over_taking': 'Overtaking other vehicles is prohibited in this area.',
                    'no overtaking': 'Overtaking is prohibited in this area.',
                    'overtaking by trucks is prohibited': 'Trucks are not allowed to overtake here.',
                    
                    // Traffic control
                    'roundabout': 'Approaching roundabout — yield and follow flow.',
                    'round-about': 'Roundabout ahead — yield and follow circular flow of traffic.',
                    'traffic light ahead': 'Traffic signal ahead — be ready to stop.',
                    'traffic_signal': 'Traffic lights ahead — be prepared to stop or slow down.',
                    
                    // Keep direction
                    'keep left': 'Stay on the left side of the road.',
                    'keep-left': 'Stay on the left side of the road or obstacle.',
                    'keep right': 'Stay on the right side of the road.',
                    'keep-right': 'Stay on the right side of the road or obstacle.',
                    
                    // Road features
                    'road narrows': 'Road ahead becomes narrower — drive cautiously.',
                    'road narrows on right': 'Right side of the road becomes narrower — slow down and keep left.',
                    'two-way traffic ahead': 'Be aware that traffic moves in both directions.',
                    'bridge ahead': 'Bridge ahead — check speed and load restrictions.',
                    'tunnel ahead': 'Tunnel coming up — turn on lights if needed.',
                    
                    // Warnings
                    'stop sign ahead': 'Prepare to stop ahead.',
                    'attention please-': 'Pay attention — potential danger or special instructions ahead.',
                    'dangerous left curve ahead': 'Sharp left curve ahead — reduce speed and navigate carefully.',
                    'dangerous rright curve ahead': 'Sharp right curve ahead — reduce speed and navigate carefully.',
                    'left zig zag traffic': 'Road curves left and right ahead — reduce speed and stay alert.',
                    
                    // Restrictions
                    'no trucks': 'Trucks are prohibited beyond this point.',
                    'truck traffic is prohibited': 'Trucks are not allowed beyond this point.',
                    
                    // Crossings
                    'animal crossing': 'Animals may cross the road — stay alert.',
                    'railway crossing': 'Railroad tracks ahead — proceed with caution.',
                    'falling rocks': 'Possible falling rocks — drive carefully.',
                    
                    // Zones
                    'school zone': 'Slow down — children may be nearby.',
                    'no horns': 'No use of horns in this area.',
                    'pedestrian zone': 'Pedestrians only — vehicles must stop or park.',
                    'bus stop': 'Bus stop ahead — watch for buses pulling out.',
                    'speed camera ahead': 'Speed check zone ahead — keep within limits.',
                    
                    // Cycle routes
                    'cycle route ahead warning': 'Cyclists may be on the road — share the road and slow down.',
                };
                
                // Try exact match first (case-insensitive)
                const lowerClass = signClass.toLowerCase().trim();
                if (descriptions[lowerClass]) {
                    return descriptions[lowerClass];
                }
                
                // Try partial match
                for (const [key, desc] of Object.entries(descriptions)) {
                    if (lowerClass.includes(key) || key.includes(lowerClass)) {
                        return desc;
                    }
                }
                
                // Default description with the sign name
                return `Pay attention to this "${signClass}" sign for important traffic information.`;
            }
            
            function displayResults(data) {
                results.innerHTML = '';
                
                if (data.predictions && data.predictions.length > 0) {
                    let html = '<div class="result-card">';
                    html += '<div class="result-title">🎯 Detected ' + data.predictions.length + ' road sign(s)</div>';
                    
                    data.predictions.forEach((pred, index) => {
                        const confidence = (pred.confidence * 100).toFixed(1);
                        const description = getSignDescription(pred.class);
                        html += `
                            <div class="detection-item">
                                <div>
                                    <span class="detection-class">${pred.class}</span>
                                    <span class="detection-confidence">${confidence}% confidence</span>
                                </div>
                                <div class="detection-description">${description}</div>
                            </div>
                        `;
                    });
                    
                    html += '</div>';
                    results.innerHTML = html;
                } else {
                    results.innerHTML = '<div class="result-card"><div class="result-title">No road signs detected in this image</div></div>';
                }
                
                results.style.display = 'block';
            }
            
            function showError(message) {
                errorDiv.textContent = '❌ Error: ' + message;
                errorDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Upload an image file (multipart/form-data).
    Returns JSON: { message, detections: [ {label, confidence, x, y, width, height, raw} ], raw: {...} }
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # call Roboflow detect endpoint
        # Use files parameter: file name, bytes, mime
        files = {"file": (file.filename or "image.jpg", io.BytesIO(image_bytes), file.content_type or "image/jpeg")}
        url = f"{ROBOFLOW_API_URL}?api_key={ROBOFLOW_API_KEY}"

        resp = requests.post(url, files=files, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as re:
        raise HTTPException(status_code=503, detail=f"Roboflow request failed: {str(re)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if resp.status_code != 200:
        # forward Roboflow error for easier debugging
        raise HTTPException(status_code=resp.status_code, detail=f"Roboflow returned {resp.status_code}: {resp.text}")

    try:
        rf_json = resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Roboflow returned non-JSON response")

    # Normalize into clean detections
    detections = normalize_roboflow_response(rf_json)

    return JSONResponse({
        "message": "Detection successful",
        "detections": detections,
        "raw": rf_json  # you can remove raw later for production
    })
