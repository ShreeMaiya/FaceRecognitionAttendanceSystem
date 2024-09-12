# Face Recognition Attendance System

This project is a **Face Recognition Attendance System** built using Python, OpenCV, and face recognition libraries. It captures real-time face data through a webcam, identifies the face, and logs attendance into a CSV file with the corresponding name, date, and time.

## Features
- Real-time face detection using a webcam.
- Automatic attendance marking in a CSV file with date and time.
- Labels for both recognized and unknown faces.
- Configurable accuracy threshold for face recognition.
- Visual feedback for users, including guidance prompts.

## How It Works
1. The system trains itself on input images provided with labels (person's name).
2. During real-time detection, the webcam captures a face and compares it with the trained data.
3. If the face is recognized, it logs the name along with the date and time in a CSV file.
4. If the face is not recognized, it displays 'Unknown' under the bounding box.
5. The system continuously monitors the webcam for new faces.

## Tech Stack
- **Python**: Main programming language used for implementing the project.
- **OpenCV**: For real-time face detection.
- **face_recognition library**: For face recognition and matching.
- **CSV Module**: For logging attendance into a CSV file.

## Installation and Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ShreeMaiya/FaceRecognitionAttendanceSystem.git
    cd FaceRecognitionAttendanceSystem
    ```

2. **Prepare Training Data**:
    - Place the images of the individuals to be recognized in a folder (e.g., `images/`).
    - Ensure that each image file is named after the individual (e.g., `John_Doe.jpg`).

3. **Run the system**:
    ```bash
    python main.py
    ```

## Usage
- **Input Training Data**: Add images to the `images/` folder.
- **Launch the Webcam**: When you run the script, the webcam will start detecting faces.
- **Attendance Logging**: Recognized faces will be logged into `attendance.csv` with the name, date, and time.


## Future Improvements
- Add a user interface for better control.
- Allow manual corrections of attendance.
- Integrate with a database for centralized logging.

## Contribution
Feel free to fork this repository and make your own contributions. Pull requests are welcome!
