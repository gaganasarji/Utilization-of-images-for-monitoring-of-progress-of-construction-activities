"""
Construction Progress Monitoring System - Backend API
Complete version with web interface
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
import json
import os

try:
    from ultralytics import YOLO
    USE_YOLO = True
except ImportError:
    USE_YOLO = False
    print("YOLO not available, using simulated detection")

app = Flask(_name_)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Construction equipment classes
EQUIPMENT_CLASSES = [
    'excavator', 'bulldozer', 'crane', 'concrete_mixer',
    'dump_truck', 'worker', 'scaffolding', 'materials',
    'safety_helmet', 'safety_vest'
]

# Initialize model
if USE_YOLO:
    try:
        model = YOLO('yolov8n.pt')
    except:
        model = None
        USE_YOLO = False
else:
    model = None

class ConstructionAnalyzer:
    def _init_(self):
        self.model = model
        self.history = []
    
    def analyze_image(self, image_path):
        """Analyze construction site image"""
        image = cv2.imread(image_path)
        
        if USE_YOLO and self.model:
            results = self.model(image)
            detections = self.parse_yolo_results(results)
        else:
            detections = self.simulate_detection(image)
        
        progress_data = self.calculate_progress(detections, image)
        safety_score = self.analyze_safety(detections)
        annotated_image = self.annotate_image(image, detections)
        
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'detections': detections,
            'progress': progress_data,
            'safety': safety_score,
            'annotated_image': self.encode_image(annotated_image)
        }
        
        self.history.append(analysis_result)
        return analysis_result
    
    def parse_yolo_results(self, results):
        """Parse YOLO detection results"""
        detections = {}
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[cls]
                
                if class_name not in detections:
                    detections[class_name] = []
                
                detections[class_name].append({
                    'confidence': conf,
                    'bbox': box.xyxy[0].tolist()
                })
        
        return detections
    
    def simulate_detection(self, image):
        """Simulate detection for demo"""
        detections = {}
        np.random.seed(hash(image.tobytes()) % (2**32))
        
        for equipment in EQUIPMENT_CLASSES:
            count = np.random.randint(0, 5)
            if count > 0:
                detections[equipment] = [
                    {
                        'confidence': 0.85 + np.random.random() * 0.15,
                        'bbox': [
                            np.random.randint(0, image.shape[1] - 100),
                            np.random.randint(0, image.shape[0] - 100),
                            np.random.randint(100, image.shape[1]),
                            np.random.randint(100, image.shape[0])
                        ]
                    }
                    for _ in range(count)
                ]
        
        return detections
    
    def calculate_progress(self, detections, image):
        """Calculate construction progress"""
        equipment_count = sum(len(v) for v in detections.values())
        worker_count = len(detections.get('worker', []))
        
        base_progress = min(100, equipment_count * 5 + worker_count * 3)
        
        progress = {
            'overall': base_progress,
            'foundation': min(100, base_progress * 1.2),
            'structure': min(100, base_progress * 0.9),
            'finishing': min(100, base_progress * 0.6),
        }
        
        return progress
    
    def analyze_safety(self, detections):
        """Analyze safety compliance"""
        total_workers = len(detections.get('worker', []))
        helmets = len(detections.get('safety_helmet', []))
        vests = len(detections.get('safety_vest', []))
        
        if total_workers == 0:
            total_workers = max(helmets, vests, 1)
        
        helmet_compliance = (helmets / total_workers * 100) if total_workers > 0 else 100
        vest_compliance = (vests / total_workers * 100) if total_workers > 0 else 100
        
        safety_score = (helmet_compliance + vest_compliance) / 2
        
        return {
            'score': safety_score,
            'total_workers': total_workers,
            'workers_with_helmets': helmets,
            'workers_with_vests': vests
        }
    
    def annotate_image(self, image, detections):
        """Draw bounding boxes"""
        annotated = image.copy()
        colors = {
            'excavator': (0, 255, 0),
            'bulldozer': (255, 0, 0),
            'crane': (0, 0, 255),
            'worker': (255, 255, 0),
            'safety_helmet': (0, 255, 255),
        }
        
        for class_name, items in detections.items():
            color = colors.get(class_name, (128, 128, 128))
            for item in items:
                bbox = item['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {item['confidence']:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def encode_image(self, image):
        """Encode image to base64"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

analyzer = ConstructionAnalyzer()

@app.route('/')
def index():
    """Serve the web interface"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Construction Monitoring API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                    background: #f5f5f5;
                }
                .container {
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #1e3c72;
                }
                .status {
                    padding: 15px;
                    background: #4CAF50;
                    color: white;
                    border-radius: 5px;
                    margin: 20px 0;
                }
                .endpoint {
                    padding: 10px;
                    background: #f9f9f9;
                    border-left: 4px solid #1e3c72;
                    margin: 10px 0;
                }
                .note {
                    background: #fff3cd;
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 4px solid #ffc107;
                    margin: 20px 0;
                }
                code {
                    background: #f5f5f5;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: monospace;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏗 Construction Monitoring API</h1>
                <div class="status">
                    <strong>✓ API is running!</strong>
                </div>
                
                <div class="note">
                    <strong>⚠ Web Interface Missing</strong><br>
                    The <code>index.html</code> file is not found. Please create it in the project folder to see the full web interface.
                </div>
                
                <h2>📡 Available Endpoints</h2>
                
                <div class="endpoint">
                    <strong>POST /api/analyze</strong><br>
                    Upload and analyze construction site image
                </div>
                
                <div class="endpoint">
                    <strong>GET /api/history</strong><br>
                    Get analysis history
                </div>
                
                <div class="endpoint">
                    <strong>GET /api/progress</strong><br>
                    Get current progress
                </div>
                
                <div class="endpoint">
                    <strong>GET /api/export</strong><br>
                    Export analysis report
                </div>
                
                <div class="endpoint">
                    <strong>GET /health</strong><br>
                    Health check
                </div>
                
                <h2>🧪 Test the API</h2>
                <p>Use curl, Postman, or any HTTP client:</p>
                <pre style="background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto;">
curl -X POST -F "image=@construction_site.jpg" http://localhost:5000/api/analyze
                </pre>
                
                <h2>📚 Documentation</h2>
                <p>Check the README.md file for detailed instructions.</p>
            </div>
        </body>
        </html>
        """

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze uploaded construction site image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        results = analyzer.analyze_image(filepath)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    return jsonify(analyzer.history)

@app.route('/api/progress', methods=['GET'])
def get_progress():
    """Get overall project progress"""
    if not analyzer.history:
        return jsonify({'progress': 0, 'analyses': 0})
    
    latest = analyzer.history[-1]
    return jsonify({
        'progress': latest['progress']['overall'],
        'analyses': len(analyzer.history),
        'latest_timestamp': latest['timestamp']
    })

@app.route('/api/export', methods=['GET'])
def export_report():
    """Export analysis report"""
    report = {
        'project_name': 'Construction Site A',
        'export_date': datetime.now().isoformat(),
        'total_analyses': len(analyzer.history),
        'history': analyzer.history
    }
    
    return jsonify(report)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': USE_YOLO,
        'timestamp': datetime.now().isoformat(),
        'total_analyses': len(analyzer.history)
    })

if _name_ == '_main_':
    print("="*60)
    print("🏗  CONSTRUCTION MONITORING API")
    print("="*60)
    print(f"YOLO Model: {'✓ Loaded' if USE_YOLO else '⚠ Simulated (Install ultralytics for real detection)'}")
    print(f"Upload folder: {UPLOAD_FOLDER}/")
    print(f"Results folder: {RESULTS_FOLDER}/")
    print("="*60)
    print()
    print("🌐 Server starting at:")
    print("   Local:   http://localhost:5000")
    print("   Network: http://0.0.0.0:5000")
    print()
    print("📡 API Endpoints:")
    print("   GET  /              - Web interface")
    print("   POST /api/analyze   - Analyze image")
    print("   GET  /api/history   - Get history")
    print("   GET  /api/progress  - Get progress")
    print("   GET  /api/export    - Export report")
    print("   GET  /health        - Health check")
    print()
    print("="*60)
    print("Press CTRL+C to quit")
    print("="*60)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)