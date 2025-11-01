# Road Sign Detection API

This is a FastAPI-based REST API that detects road signs using a pre-trained YOLO model hosted on **Roboflow**. The API includes a beautiful web UI for easy image uploads and displays helpful descriptions for detected road signs.

## 🚀 Features
- Upload an image and get detection results (bounding boxes, classes, confidence)
- Normalized detection response format for consistent data structure
- Beautiful, modern web UI with drag-and-drop support
- Helpful descriptions for each detected road sign
- Uses Roboflow's hosted model (no local GPU needed)
- Modular architecture with separate config and utility modules
- Comprehensive error handling and timeout management

## 🧩 Tech Stack
- FastAPI
- Python 3.10+

## 📁 Project Structure
roadsign-api/
├── main.py           # FastAPI app with endpoints and UI
├── config.py         # Configuration and environment variables
├── utils.py          # Utility functions for normalizing responses
├── requirements.txt  # Python dependencies
├── .env             # Environment variables (API keys, URLs)
└── README.md        # This file

## ⚙️ Setup Instructions

1. **Clone or copy this project**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env` file in the project root with the following:
   ```env
   ROBOFLOW_API_URL=https://detect.roboflow.com/your-model-id/version
   ROBOFLOW_API_KEY=your_api_key_here
   REQUEST_TIMEOUT=15
   ```

4. **Run the application**
   ```bash
   uvicorn main:app --reload
   ```

5. **Access the application**
   - Web UI: `http://localhost:8000`
   - API Docs: `http://localhost:8000/docs`
   - Health Check: `http://localhost:8000/health`

## 📡 API Endpoints

### `GET /`
Returns the web UI for uploading images and viewing detection results.

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

### `POST /detect`
Upload an image file to detect road signs.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file (JPG, PNG, JPEG)

**Response:**
```json
{
  "message": "Detection successful",
  "detections": [
    {
      "label": "stop",
      "confidence": 0.95,
      "x": 320.5,
      "y": 240.3,
      "width": 150.2,
      "height": 150.8,
      "raw": { ... }
    }
  ],
  "raw": { ... }
}
```

**Fields:**
- `label`: The detected road sign class/type
- `confidence`: Confidence score (0-1)
- `x`, `y`: Center coordinates of the bounding box
- `width`, `height`: Dimensions of the bounding box
- `raw`: Original prediction data from the model
- `raw` (top-level): Complete Roboflow API response

## 🏗️ Architecture

### `main.py`
- FastAPI application setup
- Web UI (HTML/CSS/JavaScript)
- API endpoints (`/`, `/health`, `/detect`)
- Handles file uploads and responses

### `config.py`
- Loads environment variables from `.env`
- Validates required configuration
- Exports configuration constants

### `utils.py`
- `normalize_roboflow_response()`: Normalizes various Roboflow model output formats into a consistent structure
- Handles different bounding box formats (center x/y, bbox arrays, etc.)
- Extracts labels, confidence scores, and coordinates

## 🎨 Web UI Features

- **Drag & Drop**: Drag images directly onto the upload area
- **Click to Upload**: Click the upload area to browse files
- **Image Preview**: See your uploaded image before detection
- **Detection Results**: View detected road signs with confidence scores
- **Sign Descriptions**: Get helpful explanations for each detected sign
- **Responsive Design**: Beautiful gradient UI that works on all devices
- **Error Handling**: Clear error messages for troubleshooting

## 🔧 Error Handling

The API includes comprehensive error handling:
- **400**: Empty file uploaded
- **503**: Roboflow API connection failed
- **502**: Invalid response from Roboflow
- **500**: Internal server error

## 📝 Example Usage

### Using cURL
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

### Using Python
```python
import requests

url = "http://localhost:8000/detect"
files = {"file": open("road_sign.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## 🚦 Supported Road Signs

The model can detect various road signs including:
- Stop signs
- Yield/Give way signs
- Speed limit signs
- No entry signs
- Directional signs (turn left/right, one way, etc.)
- Warning signs (pedestrian crossing, school zone, etc.)
- Parking signs
- And many more!

## 🚀 Deployment

### Push to GitHub

1. **Initialize Git repository** (if not already done)
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Road Sign Detection API"
   ```

2. **Create a new repository on GitHub**
   - Go to [GitHub](https://github.com/new)
   - Create a new repository (e.g., `roadsign-api`)
   - Don't initialize with README (you already have one)

3. **Push to GitHub**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/roadsign-api.git
   git branch -M main
   git push -u origin main
   ```

### Deploy on Render

#### Option 1: Using Blueprint (Recommended)

1. **Push your code to GitHub** (follow steps above)

2. **Go to [Render Dashboard](https://dashboard.render.com/)**

3. **Click "New" → "Blueprint"**

4. **Connect your GitHub repository**
   - Authorize Render to access your GitHub account
   - Select the `roadsign-api` repository

5. **Render will automatically detect `render.yaml`**

6. **Set Environment Variables**
   - `ROBOFLOW_API_URL`: Your Roboflow API URL
   - `ROBOFLOW_API_KEY`: Your Roboflow API key
   - `REQUEST_TIMEOUT`: 15 (already set in render.yaml)

7. **Click "Apply"** and wait for deployment to complete

#### Option 2: Manual Web Service

1. **Go to [Render Dashboard](https://dashboard.render.com/)**

2. **Click "New" → "Web Service"**

3. **Connect your GitHub repository**

4. **Configure the service:**
   - **Name**: `roadsign-api`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

5. **Add Environment Variables:**
   - `ROBOFLOW_API_URL`: Your Roboflow API URL
   - `ROBOFLOW_API_KEY`: Your Roboflow API key
   - `REQUEST_TIMEOUT`: `15`

6. **Click "Create Web Service"**

7. **Wait for deployment** (usually takes 2-5 minutes)

8. **Access your app** at the provided Render URL (e.g., `https://roadsign-api.onrender.com`)

### Important Notes for Deployment

- ✅ The `.env` file is in `.gitignore` and won't be pushed to GitHub
- ✅ Use `.env.example` as a template for required environment variables
- ✅ Always set environment variables in Render's dashboard
- ⚠️ Free tier on Render may spin down after inactivity (takes ~30s to wake up)
- ⚠️ Make sure your Roboflow API key has sufficient credits/quota

## 📄 License

This project is a wrapper for Roboflow's API. Please refer to Roboflow's terms of service for API usage guidelines.
