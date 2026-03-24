# AI Focus Detection System

A real-time attention detection and interaction system using computer vision and facial landmark analysis.

## Overview

This project implements a lightweight attention detection pipeline using webcam input.  
It estimates user focus based on facial behavior signals and integrates a simple interactive feedback system.

## Features

- Real-time webcam processing
- Facial landmark detection (MediaPipe)
- Head orientation analysis (left/right/forward)
- Looking down detection (proxy for distraction)
- Eye Aspect Ratio (EAR) for drowsiness detection
- Temporal smoothing for stable predictions
- Attention-driven scoring system (interactive demo)

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy

## Methodology

The system extracts facial keypoints and computes:

- Head pose approximation via landmark geometry
- Eye Aspect Ratio (EAR) for eye state detection
- Face scale for distance estimation

Frame-level predictions are stabilized using multi-frame aggregation to reduce noise.

## Demo Logic

- Focused → score increases
- Distracted / Drowsy → score decreases
- Sustained attention required to "win"

## Installation

```bash
pip install -r requirements.txt
