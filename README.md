# Adaptive Traffic Signal Control

A computer vision based traffic signal system that adjusts green signal timing according to the traffic detected in each lane.

The project uses YOLOv8 for vehicle detection along with OpenCV-based image processing to handle shadows, improve the input frames, and calculate lane-wise traffic density. A Tkinter GUI is used to display the live detection results and traffic analytics.

## What the project does

In a fixed-time traffic signal system, every lane gets a predetermined amount of green time. This can result in one lane having very little traffic while another lane has a long queue.

This project tries to solve that problem by measuring the traffic in each lane and allocating green time according to the current traffic conditions.

The system considers:

- Number of vehicles
- Lane occupancy
- Waiting time
- Vehicle velocity
- Overall lane priority

The calculated priority is then used to distribute the available green time between lanes.

## Main Features

### Vehicle Detection

YOLOv8 is used to detect vehicles from the incoming video feed.

The current implementation uses:

- YOLOv8s
- OpenCV
- Ultralytics
- Python

### Shadow Removal

Shadows can affect vehicle detection and lead to incorrect traffic counts.

To reduce this problem, the project uses HSV-based processing followed by morphological operations such as erosion, dilation and closing.

### Traffic Density Calculation

For each lane, the system calculates traffic-related metrics such as:

- Vehicle count
- Pixel occupancy
- Waiting time
- Vehicle velocity

These values are smoothed before being used by the signal control logic.

### Adaptive Green-Time Allocation

The system calculates a priority value for each lane and distributes green time proportionally.

Minimum and maximum green-time limits are used so that a lane is not ignored for too long.

If detection confidence becomes too low, the system can fall back to a fixed-time schedule.

### GUI

The Tkinter interface provides:

- Live video with vehicle detection
- Lane-wise traffic information
- FPS information
- Traffic analytics
- Signal timing information
- Adjustable parameters

## How the System Works

1. **Video Input**  
   The system takes a traffic video as input.

2. **Frame Preprocessing**  
   OpenCV is used to preprocess the frames. Filtering and image-processing techniques are applied to improve the input before detection.

3. **Shadow Removal**  
   HSV-based processing is used to reduce the effect of shadows. Morphological operations are then applied to improve the processed image.

4. **Vehicle Detection**  
   YOLOv8 is used to detect vehicles in each frame.

5. **Traffic Analysis**  
   The detected vehicles are analyzed lane by lane. The system calculates traffic-related values such as vehicle count, occupancy, waiting time and velocity.

6. **Lane Priority Calculation**  
   The traffic information is combined to determine the priority of each lane.

7. **Adaptive Signal Timing**  
   Green time is allocated based on the calculated lane priorities, while minimum and maximum limits are maintained.

8. **Traffic Visualization**  
   The GUI displays the vehicle detections, lane information, traffic metrics and signal timing.

## Project Structure

```text
adaptive-traffic-signal-control/
│
├── static/
├── templates/
│
├── app.py
├── gui.py
├── main.py
├── utils.py
│
├── requirements.txt
├── README.md
└── yolov8s.pt
```

Installation
1. Clone the repository
git clone https://github.com/YOUR-USERNAME/adaptive-traffic-signal-control.git
2. Move into the project directory
cd adaptive-traffic-signal-control
3. Install the required packages
pip install -r requirements.txt

The project was developed using Python and uses the YOLOv8 model included in the repository.

Running the Project

Run the main application:

python main.py

The application can then be used with the available traffic video inputs. The GUI displays the vehicle detection results and traffic analytics.

Results

The project was tested using different combinations of image preprocessing techniques and YOLOv8 detection.

The best-performing configuration was the combination of morphological processing, HSV shadow removal, and YOLOv8.

Configuration	Detection Accuracy	False Positive Rate	FPS	Wait Time Reduction
Gaussian Blur + YOLOv8	88%	25%	15	15%
Morphological Filtering + YOLOv8	91%	20%	17	20%
HSV Shadow Removal + YOLOv8	93%	12%	18	22%
Morphology + HSV + YOLOv8	95%	7%	20	25%

Note: These results are based on the project's test scenarios and should not be considered production-level performance.

Limitations

The current implementation has a few limitations:

Detection performance can vary depending on camera angle and video quality.
Heavy traffic and vehicle occlusion can affect detection accuracy.
Shadow removal depends on the selected HSV threshold values.
The current system is primarily designed for lane-wise video inputs.
Real-world deployment would require testing with live traffic data and actual traffic signal hardware.
Future Scope

The project can be extended in several ways:

Fine-tune the YOLOv8 model using local traffic datasets.
Improve vehicle tracking across consecutive frames.
Use reinforcement learning for longer-term traffic prediction and signal optimization.
Coordinate traffic signals across multiple intersections.
Combine camera-based detection with IoT sensors.
Deploy the system on edge devices for real-time processing.
Add cloud-based traffic monitoring and data logging.
