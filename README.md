# ExamEye – AI-Based Exam Monitoring System

ExamEye is an AI-powered exam monitoring system that detects suspicious student behavior during exams using computer vision and real-time pose analysis.

This project was developed as a computer vision and AI monitoring solution for academic environments.

The system uses a webcam or recorded video to monitor students, track body movements, and automatically flag suspicious actions such as repeated head turning or looking down for long periods (possible hidden phone usage).

# Features

* Real-time student detection and tracking
* Head turn detection
* Head-down / possible phone usage detection
* Multi-person tracking using ByteTrack
* Automatic screenshot capture
* Automatic suspicious video clip saving
* SQLite event logging database
* Streamlit monitoring dashboard
* Automatic PDF report generation

# Technologies Used

* Python
* OpenCV
* YOLOv8 Pose
* Ultralytics YOLO
* ByteTrack
* Streamlit
* SQLite
* ReportLab

# Project Structure

```text
app/
│
├── main.py
├── generate_report.py

dashboard/
│
├── streamlit_app.py

database/
screenshots/
clips/
reports/
models/

requirements.txt
```

# Installation

Clone the repository:

```bash
git clone https://github.com/MohamadMajed771/ExamEye-AI-Monitoring-System.git
cd ExamEye-AI-Monitoring-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the YOLO Pose model:

`yolov8n-pose.pt`

Place it inside:

```text
models/
```

# Run the Project

Start the monitoring system:

```bash
py app/main.py
```

Run the dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

Generate PDF report:

```bash
py app/generate_report.py
```

# System Capabilities

* Detect suspicious behavior in real time
* Save screenshots and video evidence
* Generate monitoring reports
* Display suspicious events in a dashboard

# Future Improvements

* Eye gaze tracking
* Seat mapping
* Live classroom analytics
* Web deployment
* Teacher authentication system
* Behavioral scoring system

# Author

Mohamad Majed
Software Engineering Student
