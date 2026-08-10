# Traffic Control System - Web Application

A web-based traffic control and optimization platform using OpenCV and YOLO for vehicle detection, congestion analysis, and intelligent traffic light timing.

## Features

- **Real-time Vehicle Detection**: Uses YOLOv8 for accurate vehicle detection
- **Traffic Congestion Analysis**: Calculates density and identifies congested roads
- **Dynamic Traffic Light Control**: Optimizes green light timing based on traffic density
- **ML Performance Metrics**: Tracks accuracy, precision, recall, and F1 scores
- **Web Interface**: Modern, responsive UI accessible from any browser
- **Video/Image Processing**: Upload and process traffic footage

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download YOLO model (if not present):
```bash
# The app will try to use yolov8s.pt by default
# Make sure you have one of: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt
```

## Usage

1. Start the web server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Use the interface:
   - Click "Load Video/Image" to upload traffic footage
   - Adjust confidence threshold slider (0.1 - 0.9)
   - Click "Start" to begin processing
   - View real-time metrics and analytics

## Metrics Explained

### Traffic Metrics
- **Vehicles**: Number of detected vehicles per road
- **Density %**: Traffic density based on vehicle area coverage
- **Green Time**: Calculated optimal green light duration
- **Wait Time**: Estimated wait time for other roads
- **Accuracy %**: Detection confidence score

### ML Performance Metrics
- **Precision**: Ratio of correct detections to total detections
- **Recall**: Ratio of detected vehicles to actual vehicles
- **F1 Score**: Harmonic mean of precision and recall

### Traffic Solution
- **Priority Road**: Road with highest congestion (gets longest green time)
- **Congested Roads**: Roads with density > 50%

## How It Works

1. **Detection**: YOLO model detects vehicles in each frame
2. **Road Segmentation**: Frame is divided into multiple roads
3. **Density Calculation**: Vehicle area / road area × 100
4. **Priority Calculation**: Based on density, wait time, and vehicle count
5. **Green Time Optimization**: Dynamic allocation (8-90 seconds)
6. **Metrics Tracking**: Real-time accuracy and performance monitoring

## API Endpoints

- `GET /`: Main web interface
- `POST /api/load_model`: Load YOLO model
- `POST /api/process_frame`: Process single frame
- `GET /api/stats`: Get current statistics

## Configuration

Edit `app.py` to modify:
- Number of roads (default: 2)
- Green time range (default: 30-90s)
- Confidence threshold (default: 0.4)
- Detection classes (car, truck, bus, motorcycle, bicycle)

## Browser Compatibility

- Chrome/Edge (recommended)
- Firefox
- Safari

## Performance Tips

- Use smaller YOLO models (yolov8n.pt) for faster processing
- Lower confidence threshold for more detections
- Reduce video resolution for better performance

## Troubleshooting

**Model not loading:**
- Ensure YOLO model file exists in the same directory
- Check file permissions

**Slow processing:**
- Use yolov8n.pt instead of larger models
- Reduce video resolution
- Increase frame skip rate

**No detections:**
- Lower confidence threshold
- Check video quality and lighting
- Verify vehicle classes in configuration
