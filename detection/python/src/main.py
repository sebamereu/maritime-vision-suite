# main.py
import config
import cv2
from fps import compute_fps
from video_loader import open_video


def main():
    cap = open_video(str(config.VIDEO_PATH))

    prev_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        fps = compute_fps(prev_time, current_time)
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
