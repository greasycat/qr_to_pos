from qrdet import QRDetector
import cv2

detector = QRDetector(model_size='s')
image = cv2.imread(filename='qrs.png')
detections = detector.detect(image=image, is_bgr=True)

# Draw the detections
for detection in detections:
    print(detection)
    x1, y1, x2, y2 = detection['bbox_xyxy'] # type: ignore
    confidence = detection['confidence']
    # Ensure coordinates are standard Python ints (some detectors may give numpy types)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # type: ignore
    # Draw label with confidence (position it safely within the image)
    label = f'{confidence:.2f}'
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    label_y = max(y1, label_size[1] + 10)
    cv2.rectangle(image, (x1, label_y - label_size[1] - 10), (x1 + label_size[0], label_y), (0, 255, 0), cv2.FILLED) # type: ignore
    cv2.putText(image, label, (x1, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # type: ignore
# Save the results
cv2.imwrite(filename='qrs_detections.png', img=image) # type: ignore