# AI Focus-Controlled Shooting Game

A real-time computer vision system where your **head movement controls a player** and your **focus triggers shooting**.

---

## Demo

<img width="1278" height="723" alt="image" src="https://github.com/user-attachments/assets/77245b5e-fbe8-4fa1-87e5-0db09fe9c3b3" />


## What is this?

This project uses webcam-based facial landmark detection to infer user attention and convert it into real-time game control.

- Turn your head → move left/right  
- Stay focused → shoot bullets  
- Hit enemies → gain score  
- Reach target score → win  

---

## Why this project is interesting

This project demonstrates how computer vision signals can be used for real-time human-computer interaction, going beyond passive detection into active control.

It combines:

- Computer vision (facial landmarks)
- Behavioral inference (attention & head movement)
- Real-time interaction (game control loop)

---

## Features

- Real-time webcam processing
- Facial landmark detection (MediaPipe)
- Head orientation analysis for left/right movement
- Looking down detection (proxy for distraction)
- Eye Aspect Ratio (EAR) for drowsiness detection
- Temporal smoothing for stable predictions
- Head-controlled player movement
- Focus-driven shooting mechanism
- Real-time interactive shooting gameplay

---

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy

---

## Methodology

The system extracts facial keypoints and computes:

- Head orientation using landmark geometry
- Eye Aspect Ratio (EAR) for eye-state estimation
- Face scale for distance awareness

Frame-level predictions are stabilized using multi-frame aggregation to reduce noise.

---

## Gameplay Logic

- Turn head left/right → move player
- Stay focused → fire bullets
- Hit enemies → gain score
- Reach target score → win

---

## Installation

```bash
pip install -r requirements.txt
