import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load the saved model
try:
    model = load_model('ASL_classification.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define class names
class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['next', 'del', 'space']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV webcam
cap = cv2.VideoCapture(0)

# Tracking variables
current_letter = ""
sentence = ""
last_prediction = None
current_streak_start_time = None
streak_threshold = 0.5  # seconds to confirm a gesture
ready_for_next = False

# Reset detection variables
pending_reset = False
last_confirmed_action = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    h, w, _ = frame.shape
    predicted_letter = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand landmarks
            landmarks = []
            wrist_x, wrist_y = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
            for landmark in hand_landmarks.landmark:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                rel_x, rel_y = cx - wrist_x, cy - wrist_y
                landmarks.extend([rel_x, rel_y])

            # Predict gesture
            landmarks_np = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(landmarks_np, verbose=0)
            predicted_class = np.argmax(prediction)
            predicted_letter = class_names[predicted_class]

            # Process prediction
            if predicted_letter == last_prediction:
                if current_streak_start_time is None:
                    current_streak_start_time = time.time()
                elif time.time() - current_streak_start_time > streak_threshold:
                    # Gesture confirmed
                    if predicted_letter != "next":
                        # Update current letter
                        if predicted_letter != current_letter:
                            current_letter = predicted_letter
                            ready_for_next = True
                            pending_reset = False  # New gesture breaks reset sequence
                    else:
                        # Process confirmed 'next'
                        if ready_for_next and current_letter not in ["next", "", None]:
                            # Handle the current letter
                            if current_letter == "space":
                                sentence += " "
                            elif current_letter == "del":
                                sentence = sentence[:-1] if sentence else ""
                                # Check for consecutive deletes
                                if last_confirmed_action == "del":
                                    pending_reset = True
                                else:
                                    pending_reset = False
                            else:
                                sentence += current_letter
                                pending_reset = False
                            
                            # Update last confirmed action
                            last_confirmed_action = current_letter
                            
                            # Execute pending reset if we have two consecutive deletes
                            if pending_reset and current_letter == "del":
                                print("Double delete detected - resetting sentence")
                                sentence = ""
                                pending_reset = False
                            
                            print(f"Sentence: {sentence}")
                            
                            # Reset for next gesture
                            current_letter = ""
                            ready_for_next = False
                            current_streak_start_time = None
            else:
                # New prediction detected
                last_prediction = predicted_letter
                current_streak_start_time = time.time()
    else:
        # No hand detected
        last_prediction = None
        current_streak_start_time = None

    # Display
    cv2.putText(frame, f'Current: {current_letter}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Sentence: {sentence}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('ASL Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()