import streamlit as st
import cv2
import numpy as np
import os
from calibration_utils import calibrate_camera, undistort
from binarization_utils import binarize
from perspective_utils import birdeye
from line_utils import get_fits_by_sliding_windows, draw_back_onto_the_road, Line, get_fits_by_previous_fits
from globals import xm_per_pix, time_window
import tempfile
from tqdm import tqdm

# Cache calibration to avoid recomputing
@st.cache_resource
def load_calibration(calib_images_dir='camera_cal'):
    return calibrate_camera(calib_images_dir)

# Initialize global variables
def init_globals():
    global processed_frames, line_lt, line_rt
    processed_frames = 0
    line_lt = Line(buffer_len=time_window)
    line_rt = Line(buffer_len=time_window)

# ---------------- PIPELINE FUNCTIONS ----------------
def prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter):
    h, w = blend_on_road.shape[:2]
    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)
    off_x, off_y = 20, 15

    # Create semi-transparent overlay for thumbnails
    overlay = blend_on_road.copy()
    cv2.rectangle(overlay, (0, 0), (w, thumb_h+2*off_y), (0,0,0), -1)
    blend_on_road = cv2.addWeighted(overlay, 0.3, blend_on_road, 0.7, 0)

    # Create thumbnails
    thumb_binary = np.dstack([cv2.resize(img_binary, (thumb_w, thumb_h))]*3)
    thumb_birdeye = np.dstack([cv2.resize(img_birdeye, (thumb_w, thumb_h))]*3)
    thumb_img_fit = cv2.resize(img_fit, (thumb_w, thumb_h))

    # Place thumbnails
    blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary
    blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w), :] = thumb_birdeye
    blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit

    # Add text information
    mean_curvature = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, f'Radius: {mean_curvature:.0f}m', (860, 60), font, 0.9, (255,255,255), 2)
    cv2.putText(blend_on_road, f'Offset: {offset_meter:.2f}m', (860, 130), font, 0.9, (255,255,255), 2)

    return blend_on_road

def compute_offset_from_center(line_lt, line_rt, frame_width):
    if line_lt.detected and line_rt.detected:
        # Calculate lane center
        left_fitx = line_lt.best_fit[0] * line_lt.all_y**2 + line_lt.best_fit[1] * line_lt.all_y + line_lt.best_fit[2]
        right_fitx = line_rt.best_fit[0] * line_rt.all_y**2 + line_rt.best_fit[1] * line_rt.all_y + line_rt.best_fit[2]
        
        lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
        car_center = frame_width / 2
        offset_meter = (car_center - lane_center) * xm_per_pix
    else:
        offset_meter = -1
    return offset_meter

def process_pipeline(frame, keep_state=True):
    global processed_frames, line_lt, line_rt
    
    # Undistort frame
    img_undistorted = undistort(frame, mtx, dist, verbose=False)
    
    # Binarize frame
    img_binary = binarize(img_undistorted, verbose=False)
    
    # Get bird's eye view
    img_birdeye, M, Minv = birdeye(img_binary, verbose=False)
    
    # Detect lane lines
    if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        line_lt, line_rt, img_fit = get_fits_by_previous_fits(img_birdeye, line_lt, line_rt, verbose=False)
    else:
        line_lt, line_rt, img_fit = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=9, verbose=False)
    
    # Calculate offset
    offset_meter = compute_offset_from_center(line_lt, line_rt, frame.shape[1])
    
    # Draw lane lines back to original image
    blend_on_road = draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state)
    
    # Prepare output frame
    blend_output = prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter)
    
    processed_frames += 1
    return blend_output

# ---------------- STREAMLIT UI ----------------
st.title("Advanced Lane Detection 🚗")
st.write("Upload a video or use the sample video to detect lanes in real-time.")

# Video source selection
option = st.radio("Select Video Option", ["Upload Video", "Use Sample Video"])

# Load calibration parameters (cached)
ret, mtx, dist, rvecs, tvecs = load_calibration()

# Initialize globals
init_globals()

# Video processing
video_file = None
if option == "Upload Video":
    video_file = st.file_uploader("Upload your video (MP4)", type=["mp4"])
elif option == "Use Sample Video":
    video_file = "project_video.mp4"

if video_file:
    # Create temporary files
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    
    if option == "Upload Video":
        temp_input.write(video_file.read())
        video_path = temp_input.name
    else:
        video_path = video_file
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (frame_width, frame_height))
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    st.write("Processing video, please wait... ⏳")
    
    # Process video frames
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = process_pipeline(frame, keep_state=True)
            out.write(processed_frame)
            
            # Update progress
            progress = min(cap.get(cv2.CAP_PROP_POS_FRAMES) / total_frames, 1.0)
            progress_bar.progress(progress)
            pbar.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    
    # Display processed video
    st.video(temp_output.name)
    st.success("Processing Complete ✅")
    
    # Provide download link
    with open(temp_output.name, 'rb') as f:
        st.download_button(
            label="Download Processed Video",
            data=f.read(),
            file_name="processed_lane_detection.mp4",
            mime="video/mp4"
        )
