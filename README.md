# VigilX: A Multi-Layered Identity Security Platform

![Bank of Baroda](https://img.shields.io/badge/Bank%20of%20Baroda-Security%20Platform-orange)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-green)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow)

VigilX is a comprehensive web-based prototype designed to showcase a next-generation, multi-layered security framework for the banking industry. It addresses critical vulnerabilities in modern digital identity verification, including deepfake threats and session hijacking, by implementing a Zero Trust philosophy through several innovative features.

This repository contains the frontend code for three distinct verification modules and a unified admin monitoring dashboard.

## ✨ Core Features

The prototype is divided into two main views: a User Verification View for end-user challenges and a Bank Admin View for monitoring.

### 1. AI-Adjudicated Liveness (Video KYC)
A deepfake-resistant V-KYC method that goes beyond simple checks.

- **Unpredictable Physical Challenges**: Users are given random, complex physical instructions (e.g., "Smile while pointing to your chin").
- **AI Verification**: A key frame from a short video of the user performing the action is sent to the Google Gemini API, which uses its advanced multimodal reasoning to determine if a real human correctly performed the task.

### 2. Physiological Proof-of-Life
A cutting-edge biometric verification method to confirm a user is a living being.

- **Contactless Heart Rate Monitoring**: Uses a technique called Remote Photoplethysmography (rPPG) to measure a user's heart rate in real-time from their video stream by analyzing subtle color changes on their skin.
- **Definitive Liveness Signal**: A deepfake is a digital puppet and has no pulse. This feature provides a definitive "proof-of-life" that digital fabrications cannot spoof.
- **Real-time Streaming**: Utilizes WebSockets for high-speed, frame-by-frame analysis on a backend server.

### 3. The Identity Rhythm Monitor (Admin Dashboard)
A sophisticated command center for the bank's security team to monitor user risk continuously.

- **Unified Hybrid View**: Correlates and visualizes user activity from both on-premise and cloud systems into a single "Identity Rhythm" chart.
- **Continuous Risk Scoring**: Automatically flags anomalies and high-risk events, providing a dynamic risk score for each user.
- **On-Demand Verification**: Empowers admins to trigger a live Behavioral Authentication challenge for any suspicious user, testing their Typing Cadence, Mouse Dynamics, and Device Fingerprint against their established baseline in real-time.

## 🛠️ Technology Stack

### Frontend
- **Languages**: HTML5, CSS3, JavaScript (ES6+)
- **Styling**: Tailwind CSS
- **Data Visualization**: Chart.js
- **Real-time Communication**: Socket.IO Client

### Backend (Required for Physiological Liveness feature)
- **Language**: Python
- **Framework**: FastAPI with Python-SocketIO
- **Libraries**: OpenCV, NumPy, SciPy (for rPPG signal processing)

## 🚀 Getting Started

This prototype is a single HTML file that can be run directly in a browser.

### Prerequisites
- A modern web browser (Google Chrome is recommended)
- Visual Studio Code with the Live Server extension installed

### Running the Frontend
1. Open the project folder in VS Code
2. Right-click on the `index.html` file
3. Select "Open with Live Server" from the context menu
4. Your browser will automatically open the application

## ⚠️ Important Notes

### Physiological Verification Backend
To test the "Physiological Verification" feature, you must have the corresponding Python backend server running. The frontend is configured to connect to this server via WebSockets to stream and analyze video frames. Without the backend, this specific feature will not be able to connect.

### Gemini API Key Security
The "Video KYC Liveness Check" feature contains a hardcoded API key in the JavaScript for demonstration purposes.

