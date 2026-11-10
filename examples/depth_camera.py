from pyorbbecsdk import (
    AlignFilter,
    Config,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBSensorType,
    OBStreamType,
    Pipeline,
)
import cv2
import numpy as np
import urllib.request
import os

def download_face_detector_model():
    model_path = "face_detection_yunet_2023mar.onnx"
    if not os.path.exists(model_path):
        print("Downloading YuNet face detection model...")
        # Updated valid raw GitHub URL from the official OpenCV Zoo
        url = "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        urllib.request.urlretrieve(url, model_path)
    return model_path

def get_face_distance_mm(depth_data, depth_scale, box):
    """Median depth (in mm) over the central region of a face bounding box."""
    if depth_data is None:
        return None
    x, y, w, h = box
    # Sample the central 50% of the box to avoid background pixels at the edges
    cx0 = max(x + w // 4, 0)
    cy0 = max(y + h // 4, 0)
    cx1 = min(x + (3 * w) // 4, depth_data.shape[1])
    cy1 = min(y + (3 * h) // 4, depth_data.shape[0])
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    region = depth_data[cy0:cy1, cx0:cx1]
    valid = region[region > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * depth_scale

def main():
    # Initialize the Orbbec pipeline
    pipeline = Pipeline()
    config = Config()

    try:
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)

        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        # Only deliver frame sets that contain both color and depth
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

        pipeline.start(config)
    except Exception as e:
        print(f"Failed to start the Orbbec pipeline: {e}")
        return

    # Align depth frames into the color camera's viewpoint (D2C)
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    # Get a sample frame to initialize the YuNet face detector dimensions
    width, height = 640, 480
    model_path = download_face_detector_model()
    
    face_detector = cv2.FaceDetectorYN.create(
        model_path, "", (width, height), 0.9, 0.3, 5000
    )

    print("Gemini 330 face tracking active. Press 'q' in the video window to exit.")

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if not frames:
                continue

            # Project depth into the color camera view so pixels correspond 1:1
            aligned = align_filter.process(frames)
            if aligned:
                frames = aligned

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            depth_frame = frames.get_depth_frame()
            depth_data = None
            depth_scale = 1.0
            if depth_frame:
                try:
                    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                        (depth_frame.get_height(), depth_frame.get_width())
                    )
                    depth_scale = depth_frame.get_depth_scale()
                except ValueError:
                    depth_data = None

            w = color_frame.get_width()
            h = color_frame.get_height()
            data = np.asanyarray(color_frame.get_data())

            if color_frame.get_format() == OBFormat.RGB:
                frame = cv2.cvtColor(data.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
            elif color_frame.get_format() == OBFormat.MJPG:
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            else:
                frame = data.reshape((h, w, 3))

            # Update detector size if frame dimensions differ
            face_detector.setInputSize((frame.shape[1], frame.shape[0]))

            # Distance sampling requires depth aligned to color resolution
            if depth_data is not None and depth_data.shape != frame.shape[:2]:
                depth_data = None
            
            # Detect faces
            _, faces = face_detector.detect(frame)

            if faces is not None:
                for face in faces:
                    # Bounding box coordinates: x, y, w, h
                    box = list(map(int, face[:4]))
                    cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

                    label = "Face Tracked"
                    distance_mm = get_face_distance_mm(depth_data, depth_scale, box)
                    if distance_mm is not None:
                        label = f"Face at {distance_mm / 1000.0:.2f} m"
                    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Gemini 330 Face Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()