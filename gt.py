import cv2
import torch
import csv
import numpy as np
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Color assumption based on RGB ranges
def assume_color_name(rgb):
    r, g, b = rgb
    if r > 200 and g < 100 and b < 100:
        return 'Red'
    elif r < 100 and g > 200 and b < 100:
        return 'Green'
    elif r < 100 and g < 100 and b > 200:
        return 'Blue'
    elif r > 200 and g > 200 and b < 100:
        return 'Yellow'
    elif r > 200 and g < 100 and b > 200:
        return 'Magenta'
    elif r < 100 and g > 200 and b > 200:
        return 'Cyan'
    elif r > 200 and g > 200 and b > 200:
        return 'White'
    elif r < 100 and g < 100 and b < 100:
        return 'Black'
    elif r > 200 and g > 200 and b < 100:
        return 'Orange'
    elif r > 200 and g < 100 and b > 200:
        return 'Purple'
    elif r < 100 and g > 200 and b > 100:
        return 'Turquoise'
    elif r > 100 and g > 200 and b < 100:
        return 'Lime'
    elif r > 100 and g < 100 and b > 100:
        return 'Violet'
    elif r < 100 and g < 200 and b > 100:
        return 'Teal'
    elif r > 100 and g < 100 and b < 200:
        return 'Maroon'
    elif r < 200 and g > 100 and b < 100:
        return 'Olive'
    elif r < 200 and g < 100 and b > 100:
        return 'Navy'
    elif r > 150 and g > 100 and b < 100:
        return 'Brown'
    elif r > 100 and g < 150 and b > 150:
        return 'Pink'
    elif r > 150 and g > 150 and b < 100:
        return 'Gold'
    elif r < 100 and g > 150 and b > 150:
        return 'Sky Blue'
    elif r > 100 and g < 150 and b < 100:
        return 'Khaki'
    elif r > 100 and g > 100 and b < 150:
        return 'Beige'
    elif r < 50 and g < 50 and b < 50:
        return 'Dark Gray'
    elif r < 100 and g < 100 and b < 150:
        return 'Dark Blue'
    elif r < 150 and g < 100 and b < 100:
        return 'Dark Red'
    elif r > 200 and g > 200 and b > 150:
        return 'Light Yellow'
    elif r > 200 and g > 150 and b > 200:
        return 'Light Pink'
    elif r > 200 and g > 150 and b < 100:
        return 'Light Orange'
    elif r > 150 and g > 200 and b < 100:
        return 'Light Lime'
    elif r > 150 and g < 100 and b > 200:
        return 'Light Purple'
    elif r < 100 and g > 200 and b > 150:
        return 'Light Turquoise'
    elif r > 100 and g > 150 and b > 200:
        return 'Light Sky Blue'
    elif r > 200 and g < 100 and b < 150:
        return 'Light Purple'
    elif r < 150 and g > 200 and b < 100:
        return 'Light Green'
    elif r < 100 and g < 200 and b > 200:
        return 'Light Blue'
    elif r < 200 and g > 100 and b > 200:
        return 'Light Magenta'
    elif r > 200 and g > 100 and b < 200:
        return 'Light Cyan'
    elif r > 100 and g < 200 and b > 100:
        return 'Light Olive'
    elif r > 100 and g > 100 and b < 200:
        return 'Light Navy'
    elif r < 200 and g < 150 and b > 150:
        return 'Lighter Pink'
    elif r < 150 and g < 200 and b > 150:
        return 'Lighter Turquoise'
    elif r < 150 and g < 150 and b > 200:
        return 'Lighter Sky Blue'
    elif r < 200 and g < 150 and b > 200:
        return 'Lighter Purple'
    elif r > 150 and g < 200 and b < 150:
        return 'Lighter Green'
    elif r > 150 and g < 150 and b > 100:
        return 'Lighter Blue'
    elif r > 100 and g > 150 and b < 150:
        return 'Lighter Cyan'
    elif r > 150 and g > 100 and b > 150:
        return 'Lighter Magenta'
    else:
        return 'Unknown'

# (Assume color function remains the same as in the previous code)

# Initialize video capture (0 for webcam, or provide video file path)
cap = cv2.VideoCapture('2.mp4')

# Variables for crash detection, speed estimation, and tracking
prev_position = {}
vehicle_speed = {}
prev_frame = None  # Initialize prev_frame
is_crash = False  # Initialize is_crash

# Open a CSV file to write vehicle information including crash status, speed, and direction
with open('vehicle_info.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Vehicle_Type', 'Color_Name', 'Confidence', 'Speed', 'Crash_Status'])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the frame
        results = model(frame)

        # Process detection results
        detections = results.pred[0]  # Get detections
        for det in detections:
            # Filter detections related to vehicles
            if det[-1] in [2, 5, 7]:  # Example classes for vehicles, adjust based on your YOLOv5 classes
                class_id = int(det[-1])
                class_name = model.names[class_id]

                # Extract confidence score
                confidence = det[4]

                # Extract bounding box coordinates
                bbox = det[:4].cpu().numpy()
                x_min, y_min, x_max, y_max = map(int, bbox)
                
                # Crop the vehicle region for color detection
                vehicle_roi = frame[y_min:y_max, x_min:x_max]

                # Perform color detection (calculate mean color)
                mean_color = vehicle_roi.mean(axis=0).mean(axis=0)  # Calculate mean color
                b, g, r = mean_color.astype(int)  # Convert mean color to integers
                
                # Assume color name based on RGB ranges
                detected_color = assume_color_name((r, g, b))

                # Vehicle tracking for speed estimation
                vehicle_id = (class_name, detected_color)
                current_position = (x_min, y_min)
                if vehicle_id in prev_position:
                    # Calculate distance between current and previous position
                    distance = np.linalg.norm(np.array(current_position) - np.array(prev_position[vehicle_id]))

                    # Calculate time difference between frames
                    time_diff = time.time() - vehicle_speed.get(vehicle_id, time.time())

                    # Calculate speed (distance / time)
                    speed = distance / time_diff if time_diff > 0 else 0
                    vehicle_speed[vehicle_id] = time.time()
                else:
                    speed = 0
                prev_position[vehicle_id] = current_position

                # Crash detection by comparing frames
                if prev_frame is not None:
                    diff = cv2.absdiff(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                                       cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if cv2.contourArea(contour) > 1000:  # Tweak this area threshold for your scenario
                            is_crash = True
                            break
                    else:
                        is_crash = False

                prev_frame = frame.copy()

                # Save vehicle information to CSV including crash status and speed
                writer.writerow([class_name, detected_color, confidence, speed, is_crash])

                # Draw bounding box and text on the frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} - {detected_color} - {confidence:.2f} - Speed: {speed:.2f} - Crash: {is_crash}",
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow('YOLOv5 Vehicle Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
