# main.py
import time
import cv2

from video_loader import open_video
import config
def main():
    VIDEO_PATH = r"C:\Users\sebam\Desktop\maritime-vision-suite\data\videos\sample.mp4"

    cap = open_video(VIDEO_PATH)
#    cap = open_video(config.VIDEO_PATH)

    prev_time = None 
    ret, frame = cap.read()
    prev_time = cv2.getTickCount() / cv2.getTickFrequency()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        fps = 1.0 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if config.SHOW_FPS:
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow(config.WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == config.EXIT_KEY:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
