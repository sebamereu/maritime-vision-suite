# main.py
import time
import cv2

from video_loader import open_video
import config

def main():
    cap = cv2.VideoCapture(r"C:\Users\sebam\Desktop\maritime-vision-suite\data\videos\sample.mp4")
    if not cap.isOpened():
        raise IOError("Cannot open video")

    prev_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        if prev_time is None:
            fps = 0.0
        else:
            delta = current_time - prev_time
            fps = 1.0 / delta if delta > 0 else 0.0

        prev_time = current_time

        if config.SHOW_FPS:
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(config.WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == config.EXIT_KEY:
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
