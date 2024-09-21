# OCR-Model-Conversion
This project demonstrates how to process videos using Optical Character Recognition (OCR) with PaddleOCR, optimized for both CPU and GPU environments. The primary goal is to convert an existing GPU-based OCR model to run efficiently on a CPU, while maintaining or improving the model's accuracy and frames per second (FPS) performance.

# File Descriptions

1. gpu.py
This file contains the implementation for running Optical Character Recognition (OCR) on a GPU using PaddleOCR. It processes video frames by recognizing and extracting text from each frame. The recognized text is displayed with bounding boxes, and performance metrics like processing time and frames per second (FPS) are calculated specifically for GPU execution.

Key Features:
Utilizes GPU acceleration for faster video processing.
Computes and logs FPS to evaluate GPU performance.
Ideal for environments with GPU support for enhanced speed.

2. cpu.py
This file includes the basic implementation of the OCR video processing pipeline for the CPU. The script reads video frames, performs text recognition using PaddleOCR, and displays the recognized text on the video. FPS and processing time are also calculated, specifically for CPU-based execution.

Key Features:
Runs OCR on the CPU, without relying on GPU resources.
Computes and logs CPU-specific FPS and processing time.
Useful for systems without GPU support or for comparing CPU and GPU performance.

3. updatedcpu.py
This file contains an optimized version of the CPU-based OCR processing script. It applies various CPU optimization techniques to minimize the gap in performance between CPU and GPU executions. The optimizations aim to maintain or improve FPS and accuracy on the CPU, making it more efficient for video processing tasks.

Key Features:
CPU-optimized OCR processing for improved speed and accuracy.
Utilizes multithreading and frame resizing to enhance CPU performance.
Maintains comparable accuracy and FPS to GPU execution, making it suitable for systems with limited or no GPU resources.

Each file provides unique insights into the differences in OCR processing between GPU and CPU environments, and the optimizations made in updatedcpu.py demonstrate how the performance gap can be bridged with careful tuning.

