import os
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import paddle
import time


print(f"Using GPU: {paddle.is_compiled_with_cuda()}")

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, enable_mkldnn=True)
paddle.device.set_device('cpu') 

# Video input and output settings
input_video_path = "C:\\Users\\Utkar\\OneDrive\\Desktop\\Tensogo\\hi.mp4"
output_video_path = "output_video_cpu_optimized.mp4"

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input video FPS: {fps}")
print(f"Video resolution: {width}x{height}")
print(f"Total frames: {frame_count}")

# Set up the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_num = 0
accuracies = []
fps_values = []


start_time = time.time()

# Loop through each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to reduce processing time
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (640, 360))  # Resize for optimization

    # OCR processing on the frame
    result = ocr.ocr(img_rgb)
    total_confidence = 0
    num_words = 0    
    
    # Check if the result is not None and contains data
    if result is not None:
        for line in result:
            if line:  # Check if the line contains any words
                for word_info in line:
                    text = word_info[-1][0]
                    confidence = word_info[-1][1]  # OCR confidence score
                    total_confidence += confidence
                    num_words += 1

                    bbox = word_info[0]
                    # Draw the bounding box
                    bbox = [(int(point[0]), int(point[1])) for point in bbox]
                    cv2.polylines(frame, [np.array(bbox)], True, (0, 255, 0), 2)

                    # Put the recognized text near the bounding box
                    cv2.putText(frame, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

    # Calculate accuracy for the current frame
    if num_words > 0:
        accuracy = total_confidence / num_words  # Average confidence score
    else:
        accuracy = 0

    accuracies.append(accuracy)
    
    frame_num += 1
    elapsed_time = time.time() - start_time
    current_fps = frame_num / elapsed_time
    fps_values.append(current_fps)   

print(f"Processing frame {frame_num}/{frame_count}, Accuracy: {accuracy:.2f}, FPS: {current_fps:.2f}")


end_time = time.time()

# Calculate the total execution time and FPS
total_time = end_time - start_time
average_fps = frame_num / total_time

print(f"Total time taken for processing: {total_time:.2f} seconds")
print(f"Average FPS during processing: {average_fps:.2f}")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed and saved to", output_video_path)