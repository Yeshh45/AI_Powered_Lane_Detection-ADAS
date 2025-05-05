import cv2
import numpy as np

def canny_edge_detector(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)

def region_of_interest(image):
    height = image.shape[0]
    polygon = np.array([
        [(100, height), (1180, height), (700, 450), (580, 450)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image, mask)

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        edges = canny_edge_detector(frame)
        cropped_edges = region_of_interest(edges)
        lines = cv2.HoughLinesP(cropped_edges, 2, np.pi / 180, 100,
                                np.array([]), minLineLength=40, maxLineGap=5)
        line_image = display_lines(frame, lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

        cv2.imshow("Lane Detection", combo_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

process_video("test_videos/road.mp4")
