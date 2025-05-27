# 🤟 Sign Language to Text Conversion using ASL Landmarks (Final Year Project)

This project presents a low-cost, real-time sign language recognition system that converts American Sign Language (ASL) gestures into written text using hand landmark data. The solution is designed to enhance communication accessibility for individuals with hearing or speech impairments by recognizing ASL letters and displaying the corresponding text on a screen.
![image](https://github.com/user-attachments/assets/bc1f4601-0002-41c1-880e-693efcc8032b)


---

## 📄 Abstract

Sign language is a critical tool for communication among individuals with hearing and speech impairments. However, a lack of understanding among non-signers creates a communication barrier. This project addresses that gap by developing a real-time gesture recognition system based on computer vision and deep learning.

Using a camera and MediaPipe, the system detects ASL hand gestures and classifies them into 29 categories (26 alphabetic letters plus `space`, `next`, and `delete`). The model was trained using landmark coordinates instead of raw image data, allowing efficient deployment on a Raspberry Pi 5 with a connected LCD screen. The system is accurate, portable, and practical for daily use.
![image](https://github.com/user-attachments/assets/bef334ed-4b12-4302-b452-7ab1f273e7b2)


---

## 📊 Dataset

The dataset consists of hand gestures from two sources:

1. **Team-collected data**: 7,800 images (100 samples per letter) captured under varied lighting and backgrounds.
2. **Public datasets**: Additional ASL samples from Kaggle to increase diversity in hand shapes and skin tones.
![image](https://github.com/user-attachments/assets/1766d1b8-acb4-4270-823c-cf277dc9f4ad)


### Classes:  
- 26 ASL letters (A–Z)  
- 3 functional gestures: `space`, `next`, `delete`  

Each sample was processed using MediaPipe to extract **21 keypoints** (x, y), resulting in **42 landmark features** per image.
![image](https://github.com/user-attachments/assets/cef8de87-757f-4992-a405-742c9b70eb4e)


---

## ✋ Hand Landmark Extraction (MediaPipe)

Instead of using raw images, MediaPipe's Hands solution was used to extract 21 landmark points (x, y) per hand in real time. This significantly reduced the amount of data and improved both speed and accuracy.

- **Input**: Webcam frame
- **Output**: 42-feature vector per hand (21 points × 2 coordinates)
- **Advantages**:
  - Lower noise and background impact
  - Lightweight and faster processing
  - Consistent across lighting and skin tones

---

## 🧠 Model Architecture

A fully connected neural network was used for gesture classification.

### Architecture Overview:
- **Input layer**: 42 features
- **Dense layer 1**: 128 units + ReLU + Dropout (30%)
- **Dense layer 2**: 64 units + ReLU + Dropout (30%)
- **Output layer**: 29 units + Softmax (for multi-class classification)
![image](https://github.com/user-attachments/assets/45a1c6a4-eaa9-48a3-b8e0-19ce87a2976a)


### Training Details:
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Epochs: 50  
- Batch size: 32  

### Output Device:
The recognized gesture is sent to a **2x16 I2C LCD** for real-time display of letters and words.
![image](https://github.com/user-attachments/assets/fd5ebf8b-d645-4c2a-be2a-be0684a18862)


---

## 📈 Results

| Metric                   | Value         |
|--------------------------|---------------|
| **Validation Accuracy**  | 98%           |
| **Test Accuracy**        | 99%           |
| **Average Inference Time** | < 120 ms   |
| **Gesture Confirmation Delay** | ~0.5s |
| **LCD Update Latency**   | < 0.3s        |

- Special gestures like **delete** and **reset** were tested successfully.
- Complex sequences like forming full words (e.g., "CAT") worked well.

---

## ⚙️ How to Run

### 1. Install Dependencies

pip install tensorflow mediapipe opencv-python numpy RPLCD

### 2. Run the Real-Time Recegnitionm

python realtime.py


## 🏆 Achievements

- 🥇 **Won 1st Place** in the Final Year Project (FYP) Competition at Qassim University - College of Computer
- ✅ Achieved **99% test accuracy** and **98% validation accuracy** across 29 ASL gesture classes
- ✅ Deployed a **real-time recognition system** on Raspberry Pi with a live LCD output
![a233377e-c17c-4234-af57-de69b6000d30](https://github.com/user-attachments/assets/29a7668d-bb9f-4bb2-a08d-aed5e1397a19)


