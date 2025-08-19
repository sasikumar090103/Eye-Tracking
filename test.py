import cv2
import mediapipe as mp
import numpy as np

video_path = r"C:\Users\Sasi\Downloads\Eye Movement detection\Eye Movement detection\C0032.MP4"
output_path = "run3.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Get video properties dynamically
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:  # fallback if FPS not detected
    fps = 30

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
if not out.isOpened():
    print("Error: Could not create video writer")
    exit()

# Only initialize FaceMesh after video is confirmed to be working
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Eye and iris landmark indexes
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
LEFT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_TOP_BOTTOM = [386, 374]


# Initialize FaceMesh after confirming capture
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    movement = "No face detected"
    screen_status = "Not Looking at Screen"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            h, w, _ = frame.shape

            def get_point(idx):
                lm = face_landmarks.landmark[idx]
                return int(lm.x * w), int(lm.y * h)

            left_eye_left = get_point(LEFT_EYE[0])
            left_eye_right = get_point(LEFT_EYE[1])
            right_eye_left = get_point(RIGHT_EYE[0])
            right_eye_right = get_point(RIGHT_EYE[1])

            left_iris = np.array([get_point(i) for i in LEFT_IRIS])
            right_iris = np.array([get_point(i) for i in RIGHT_IRIS])
            left_iris_center = np.mean(left_iris, axis=0).astype(int)
            right_iris_center = np.mean(right_iris, axis=0).astype(int)

            cv2.circle(frame, tuple(left_iris_center), 2, (0, 255, 0), -1)
            cv2.circle(frame, tuple(right_iris_center), 2, (0, 255, 0), -1)

            def eye_position(eye_left, eye_right, iris_center):
                eye_width = np.linalg.norm(np.array(eye_right) - np.array(eye_left))
                iris_to_left = np.linalg.norm(np.array(iris_center) - np.array(eye_left))
                return iris_to_left / eye_width

            def vertical_iris_ratio(top_idx, bottom_idx, iris_center):
                top = get_point(top_idx)
                bottom = get_point(bottom_idx)
                eye_height = np.linalg.norm(np.array(top) - np.array(bottom))
                iris_to_top = np.linalg.norm(np.array(iris_center) - np.array(top))
                return iris_to_top / eye_height if eye_height > 0 else 0.5

            def blink_ratio(top_idx, bottom_idx, eye_left, eye_right):
                top = get_point(top_idx)
                bottom = get_point(bottom_idx)
                eye_width = np.linalg.norm(np.array(eye_right) - np.array(eye_left))
                eye_height = np.linalg.norm(np.array(top) - np.array(bottom))
                return eye_height / eye_width

            left_ratio = eye_position(left_eye_left, left_eye_right, left_iris_center)
            right_ratio = eye_position(right_eye_left, right_eye_right, right_iris_center)
            left_vertical = vertical_iris_ratio(LEFT_EYE_TOP_BOTTOM[0], LEFT_EYE_TOP_BOTTOM[1], left_iris_center)
            right_vertical = vertical_iris_ratio(RIGHT_EYE_TOP_BOTTOM[0], RIGHT_EYE_TOP_BOTTOM[1], right_iris_center)
            left_blink = blink_ratio(LEFT_EYE_TOP_BOTTOM[0], LEFT_EYE_TOP_BOTTOM[1], left_eye_left, left_eye_right)
            right_blink = blink_ratio(RIGHT_EYE_TOP_BOTTOM[0], RIGHT_EYE_TOP_BOTTOM[1], right_eye_left, right_eye_right)

            avg_ratio = (left_ratio + right_ratio) / 2
            avg_blink = (left_blink + right_blink) / 2
            avg_vertical = (left_vertical + right_vertical) / 2

            print(f"Vertical Ratio: {avg_vertical:.2f}, Horizontal Ratio: {avg_ratio:.2f}, Blink: {avg_blink:.2f}")


            if avg_blink < 0.16:
                movement = "Blink"
                screen_status = "Not Looking at Screen"
            elif avg_vertical < 0.35:
                movement = "Looking Up"
                screen_status = "Not Looking at Screen"
            elif avg_vertical > 0.65:
                movement = "Looking Down"
                screen_status = "Not Looking at Screen"
            elif avg_ratio < 0.42:
                movement = "Looking Right"
                screen_status = "Not Looking at Screen"
            elif avg_ratio > 0.57:
                movement = "Looking Left"
                screen_status = "Not Looking at Screen"
            else:
                movement = "Looking Center"
                screen_status = "Looking at Screen"

            cv2.putText(frame, f"Eye Movement: {movement}", (30, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(frame, f"Screen Status: {screen_status}", (1500, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

    out.write(frame)




cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing completed. Video saved as:", output_path)
