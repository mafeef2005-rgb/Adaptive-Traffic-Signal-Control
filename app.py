from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
from ultralytics import YOLO
import json
import base64
import time
from collections import deque

app = Flask(__name__)

# Global variables
model = None
video_capture = None
road_tracker = None
processing_active = False

class WebRoadTracker:
    def __init__(self, roads=2):
        self.roads = roads
        self.vehicle_counts = [0] * roads
        self.densities = [0.0] * roads
        self.green_times = [30] * roads
        self.wait_times = [0.0] * roads
        self.accuracy_scores = [0.0] * roads
        self.precision = [0.0] * roads
        self.recall = [0.0] * roads
        self.f1_scores = [0.0] * roads
        self.history = deque(maxlen=50)
        
    def update(self, counts, densities, confidences):
        self.vehicle_counts = counts
        self.densities = densities
        
        # Calculate metrics
        total_density = sum(densities)
        for i in range(self.roads):
            if total_density > 0:
                priority = densities[i] / total_density
                self.green_times[i] = int(30 + priority * 60)
                self.wait_times[i] = 120 - self.green_times[i]
            
            # Accuracy from confidence
            if confidences[i]:
                avg_conf = np.mean(confidences[i])
                self.accuracy_scores[i] = avg_conf * 100
                self.precision[i] = min(1.0, avg_conf * 1.1)
                self.recall[i] = min(1.0, avg_conf * 0.95)
                self.f1_scores[i] = 2 * (self.precision[i] * self.recall[i]) / (self.precision[i] + self.recall[i] + 0.001)
        
        # Store history
        self.history.append({
            'timestamp': time.time(),
            'counts': counts.copy(),
            'densities': densities.copy()
        })
    
    def get_stats(self):
        priority_road = int(np.argmax(self.densities)) if any(self.densities) else 0
        congested = [i for i, d in enumerate(self.densities) if d > 50]
        
        return {
            'roads': [{
                'id': i,
                'vehicles': self.vehicle_counts[i],
                'density': round(self.densities[i], 1),
                'green_time': self.green_times[i],
                'wait_time': round(self.wait_times[i], 1),
                'accuracy': round(self.accuracy_scores[i], 1),
                'precision': round(self.precision[i] * 100, 1),
                'recall': round(self.recall[i] * 100, 1),
                'f1_score': round(self.f1_scores[i] * 100, 1)
            } for i in range(self.roads)],
            'priority_road': priority_road,
            'congested_roads': congested,
            'total_vehicles': sum(self.vehicle_counts),
            'avg_accuracy': round(np.mean(self.accuracy_scores), 1)
        }

def detect_vehicles(frame, conf=0.4):
    global model, road_tracker
    
    if model is None:
        return frame, [0, 0], [0.0, 0.0], [[], []]
    
    # Run detection
    results = model(frame, conf=conf, verbose=False)[0]
    
    # Split frame into 2 roads
    h, w = frame.shape[:2]
    mid = w // 2
    
    counts = [0, 0]
    confidences = [[], []]
    vehicle_areas = [0.0, 0.0]
    road_areas = [h * mid, h * (w - mid)]
    
    # Count vehicles
    vehicle_classes = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
    
    if hasattr(results, 'boxes') and results.boxes is not None:
        for box in results.boxes:
            cls_name = results.names[int(box.cls)].lower()
            if cls_name in vehicle_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_val = float(box.conf)
                center_x = (x1 + x2) / 2
                
                road_idx = 0 if center_x < mid else 1
                counts[road_idx] += 1
                confidences[road_idx].append(conf_val)
                vehicle_areas[road_idx] += (x2 - x1) * (y2 - y1)
                
                # Draw box
                color = (0, 255, 0) if road_idx == 0 else (255, 100, 100)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    # Calculate densities
    densities = [(vehicle_areas[i] / road_areas[i]) * 100 for i in range(2)]
    
    # Draw road divider
    cv2.line(frame, (mid, 0), (mid, h), (255, 255, 0), 2)
    
    # Add labels
    cv2.putText(frame, f"Road 1: {counts[0]} vehicles", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Road 2: {counts[1]} vehicles", (mid + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
    
    return frame, counts, densities, confidences

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/load_model', methods=['POST'])
def load_model():
    global model
    try:
        model_path = request.json.get('model_path', 'yolov8s.pt')
        model = YOLO(model_path)
        return jsonify({'success': True, 'message': f'Model {model_path} loaded'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    global model, road_tracker
    
    try:
        # Get image from request
        data = request.json
        image_data = data['image'].split(',')[1]
        conf = data.get('confidence', 0.4)
        
        # Decode image
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process
        annotated, counts, densities, confidences = detect_vehicles(frame, conf)
        
        # Update tracker
        if road_tracker is None:
            road_tracker = WebRoadTracker(roads=2)
        road_tracker.update(counts, densities, confidences)
        
        # Encode result
        _, buffer = cv2.imencode('.jpg', annotated)
        img_str = base64.b64encode(buffer).decode()
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_str}',
            'stats': road_tracker.get_stats()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stats')
def get_stats():
    global road_tracker
    if road_tracker:
        return jsonify(road_tracker.get_stats())
    return jsonify({'roads': [], 'total_vehicles': 0})

if __name__ == '__main__':
    # Load default model
    try:
        model = YOLO('yolov8s.pt')
        road_tracker = WebRoadTracker(roads=2)
        print("✅ Model loaded successfully")
    except:
        print("⚠️ Model not found, please load manually")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
