import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from tempfile import NamedTemporaryFile
import time

# ======================================================================
# MOCK-UP UTILITY DEFINITIONS
# ======================================================================

# Global constants
xm_per_pix = 3.7 / 700
time_window = 10

class Line:
    def __init__(self, buffer_len=10):
        self.detected = False
        self.recent_xfitted = []
        self.all_x = None
        self.all_y = None
        self.best_fit = None
        self.curvature_meter = 0
        self.buffer_len = buffer_len

    def update_line(self, x, y, fit, curvature):
        self.all_x = x
        self.all_y = y
        self.best_fit = fit
        self.curvature_meter = curvature
        self.detected = True

def calibrate_camera(calib_images_dir):
    time.sleep(1)
    st.write("Camera calibration simulated.")
    return True, np.eye(3), np.zeros((4, 1)), None, None

def undistort(img, mtx, dist, verbose=False):
    return img

def binarize(img, verbose=False):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def birdeye(img, verbose=False):
    h, w = img.shape[:2]
    M = np.eye(3)
    Minv = np.eye(3)
    if len(img.shape) == 2:
        img_birdeye = np.dstack([img]*3)
    else:
        img_birdeye = img.copy()
    return img_birdeye, M, Minv

def get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=9, verbose=False):
    h, w = img_birdeye.shape[:2]
    line_lt.update_line(np.linspace(0, h, 100), np.linspace(w*0.3, w*0.3 + 10, 100), [0, 0, 0], 600)
    line_rt.update_line(np.linspace(0, h, 100), np.linspace(w*0.7, w*0.7 + 10, 100), [0, 0, 0], 700)
    img_fit = np.zeros_like(img_birdeye)
    cv2.putText(img_fit, "Sliding Windows (MOCK)", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return line_lt, line_rt, img_fit

def get_fits_by_previous_fits(img_birdeye, line_lt, line_rt, verbose=False):
    h, w = img_birdeye.shape[:2]
    line_lt.update_line(np.linspace(0, h, 100), np.linspace(w*0.3, w*0.3 + 5, 100), [0,0,0], 650)
    line_rt.update_line(np.linspace(0, h, 100), np.linspace(w*0.7, w*0.7 - 5, 100), [0,0,0], 750)
    img_fit = np.zeros_like(img_birdeye)
    cv2.putText(img_fit, "Previous Fits (MOCK)", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return line_lt, line_rt, img_fit

def draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state):
    img = img_undistorted.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (int(w*0.3), h-50), (int(w*0.7), h), (0,255,0), cv2.FILLED)
    return img

def compute_offset_from_center(line_lt, line_rt, frame_width):
    if line_lt.detected and line_rt.detected:
        line_lt_bottom = np.mean(line_lt.all_x[-10:]) if line_lt.all_x is not None else frame_width*0.3
        line_rt_bottom = np.mean(line_rt.all_x[-10:]) if line_rt.all_x is not None else frame_width*0.7
        lane_center = (line_lt_bottom + line_rt_bottom)/2
        midpoint = frame_width/2
        offset_pix = lane_center - midpoint
        offset_meter = xm_per_pix * offset_pix
    else:
        offset_meter = 0.0
    return offset_meter

def prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter):
    h, w = blend_on_road.shape[:2]
    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio*h), int(thumb_ratio*w)
    off_x, off_y = 20, 15

    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, (0,0), (w, thumb_h+2*off_y), (0,0,0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(mask, 0.2, blend_on_road, 0.8, 0)

    thumb_binary = np.dstack([cv2.resize(img_binary,(thumb_w,thumb_h))]*3)*255
    thumb_birdeye = cv2.resize(img_birdeye,(thumb_w,thumb_h))
    thumb_img_fit = cv2.resize(img_fit,(thumb_w,thumb_h))

    end_x_thumb_3 = 3*off_x + 3*thumb_w
    if end_x_thumb_3 < w:
        blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w,:] = thumb_binary
        blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*off_x+2*thumb_w,:] = thumb_birdeye
        blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*off_x+3*thumb_w,:] = thumb_img_fit
    else:
        cv2.putText(blend_on_road, "Thumbnails skipped", (off_x, off_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

    mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, f'Curvature radius: {mean_curvature_meter:.2f}m', (860,60), font, 0.9, (255,255,255), 2)
    cv2.putText(blend_on_road, f'Offset from center: {offset_meter:.2f}m', (860,130), font, 0.9, (255,255,255),2)

    return blend_on_road

TARGET_WIDTH = 960
processed_frames = 0
line_lt = Line(buffer_len=time_window)
line_rt = Line(buffer_len=time_window)

def process_pipeline(frame, mtx, dist, keep_state=True):
    global line_lt, line_rt, processed_frames

    frame_width_original = frame.shape[1]
    if frame_width_original > TARGET_WIDTH:
        scale_factor = TARGET_WIDTH / frame_width_original
        new_dim = (TARGET_WIDTH, int(frame.shape[0]*scale_factor))
        frame_resized = cv2.resize(frame, new_dim, interpolation=cv2.INTER_LINEAR)
    else:
        frame_resized = frame

    working_frame_width = frame_resized.shape[1]

    img_undistorted = undistort(frame_resized, mtx, dist)
    img_binary = binarize(img_undistorted)
    img_birdeye, M, Minv = birdeye(img_binary)

    if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        line_lt, line_rt, img_fit = get_fits_by_previous_fits(img_birdeye, line_lt, line_rt)
    else:
        line_lt, line_rt, img_fit = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt)

    offset_meter = compute_offset_from_center(line_lt, line_rt, frame_width=working_frame_width)
    blend_on_road = draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state)
    blend_output = prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter)

    processed_frames += 1

    if frame_width_original > TARGET_WIDTH:
        blend_output = cv2.resize(blend_output, (frame_width_original, frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    return blend_output

# ======================================================================
# STREAMLIT UI
# ======================================================================

st.title("Advanced Lane Detection 🚗 (Optimized)")
st.write("Upload a video or use the sample video to detect lanes in real-time. Processing speed is optimized via frame downscaling.")

option = st.radio("Select Video Option", ["Upload Video", "Use Sample Video"])
SAMPLE_VIDEO_PATH = "project_video.mp4"
video_file = None

if option == "Upload Video":
    video_file = st.file_uploader("Upload your video (MP4)", type=["mp4"])
elif option == "Use Sample Video":
    st.write(f"Using sample video: `{SAMPLE_VIDEO_PATH}`")
    video_path = SAMPLE_VIDEO_PATH

if video_file or (option=="Use Sample Video" and os.path.exists(SAMPLE_VIDEO_PATH)):

    if option=="Upload Video":
        tfile = NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        video_path = tfile.name

    if not os.path.exists(video_path):
        st.error(f"Video file not found at '{video_path}'.")
        st.stop()

    with st.spinner("Calibrating Camera... ⏳"):
        ret, mtx, dist, rvecs, tvecs = calibrate_camera('camera_cal')
        if not ret:
            st.error("Camera calibration failed.")
            st.stop()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Could not open video file at {video_path}")
        st.stop()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.write(f"Video resolution: {frame_width}x{frame_height} @ {fps:.1f} FPS")
    st.write(f"Processing frames with max width {TARGET_WIDTH}px for speed")

    fd, temp_out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_out_path, fourcc, fps, (frame_width, frame_height))

    processed_frames = 0
    line_lt = Line(buffer_len=time_window)
    line_rt = Line(buffer_len=time_window)

    status_text = st.empty()
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)

    start_time = time.time()
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_pipeline(frame, mtx, dist, keep_state=True)
            out.write(processed_frame)

            if frame_count % 5 == 0:
                frame_placeholder.image(processed_frame, channels="BGR",use_container_width=True, caption=f"Frame {frame_count}")

            frame_count += 1
            progress_bar.progress(min(frame_count/total_frames, 1.0))
            if frame_count % 30 == 0:
                status_text.text(f"Processing frame {frame_count} of {total_frames}...")

    finally:
        cap.release()
        out.release()

    end_time = time.time()
    total_time = end_time - start_time
    status_text.text(f"Processing complete! {frame_count} frames in {total_time:.2f} seconds.")

    st.video(temp_out_path)

    st.download_button(
        label="Download Processed Video",
        data=open(temp_out_path, "rb").read(),
        file_name="processed_video.mp4",
        mime="video/mp4"
    )

    st.success("Processing Complete ✅")

    if 'tfile' in locals() and os.path.exists(tfile.name):
        os.remove(tfile.name)
    if os.path.exists(temp_out_path):
        os.remove(temp_out_path)
