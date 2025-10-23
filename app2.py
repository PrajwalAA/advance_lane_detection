import streamlit as st
import cv2
import numpy as np
import os
from calibration_utils import calibrate_camera, undistort
from binarization_utils import binarize
from perspective_utils import birdeye
from line_utils import get_fits_by_sliding_windows, draw_back_onto_the_road, Line, get_fits_by_previous_fits
from globals import xm_per_pix, time_window
from tempfile import NamedTemporaryFile
import tempfile

# Global variables
processed_frames = 0
line_lt = Line(buffer_len=time_window)
line_rt = Line(buffer_len=time_window)

# ---------------- PIPELINE FUNCTIONS ----------------
def prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter):
    h, w = blend_on_road.shape[:2]
    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)
    off_x, off_y = 20, 15

    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, (0, 0), (w, thumb_h+2*off_y), (0,0,0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(mask, 0.2, blend_on_road, 0.8, 0)

    thumb_binary = np.dstack([cv2.resize(img_binary, (thumb_w, thumb_h))]*3) * 255
    thumb_birdeye = np.dstack([cv2.resize(img_birdeye, (thumb_w, thumb_h))]*3) * 255
    thumb_img_fit = cv2.resize(img_fit, (thumb_w, thumb_h))

    blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary
    blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w), :] = thumb_birdeye
    blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit

    mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, f'Curvature radius: {mean_curvature_meter:.2f}m', (860, 60), font, 0.9, (255,255,255), 2)
    cv2.putText(blend_on_road, f'Offset from center: {offset_meter:.2f}m', (860, 130), font, 0.9, (255,255,255), 2)

    return blend_on_road

def compute_offset_from_center(line_lt, line_rt, frame_width):
    if line_lt.detected and line_rt.detected:
        line_lt_bottom = np.mean(line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
        line_rt_bottom = np.mean(line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])
        lane_width = line_rt_bottom - line_lt_bottom
        midpoint = frame_width / 2
        offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint)
        offset_meter = xm_per_pix * offset_pix
    else:
        offset_meter = -1
    return offset_meter

def process_pipeline(frame, keep_state=True):
    global line_lt, line_rt, processed_frames
    img_undistorted = undistort(frame, mtx, dist, verbose=False)
    img_binary = binarize(img_undistorted, verbose=False)
    img_birdeye, M, Minv = birdeye(img_binary, verbose=False)

    if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        line_lt, line_rt, img_fit = get_fits_by_previous_fits(img_birdeye, line_lt, line_rt, verbose=False)
    else:
        line_lt, line_rt, img_fit = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=9, verbose=False)

    offset_meter = compute_offset_from_center(line_lt, line_rt, frame_width=frame.shape[1])
    blend_on_road = draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state)
    blend_output = prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter)
    processed_frames += 1
    return blend_output

# ---------------- STREAMLIT UI ----------------
st.title("Advanced Lane Detection 🚗")
st.write("Upload a video or use the sample video to detect lanes in real-time.")

# Upload or sample selection
option = st.radio("Select Video Option", ["Upload Video", "Use Sample Video"])

video_file = None
if option == "Upload Video":
    video_file = st.file_uploader("Upload your video (MP4)", type=["mp4"])
elif option == "Use Sample Video":
    video_file = "project_video.mp4"

if video_file:
    if option == "Upload Video":
        # Save uploaded video temporarily
        tfile = NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name
    else:
        video_path = video_file

    st.write("Processing video, please wait... ⏳")
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')
    cap = cv2.VideoCapture(video_path)

    # Create temporary output file
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_out.name, fourcc, 20.0,
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_pipeline(frame, keep_state=True)
        out.write(processed_frame)

    cap.release()
    out.release()
    st.video(temp_out.name)
    st.success("Processing Complete ✅")
