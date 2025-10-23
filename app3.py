import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from tempfile import NamedTemporaryFile
import time

# ======================================================================
# MOCK-UP UTILITY DEFINITIONS (TO MAKE THE CODE SELF-CONTAINED AND RUNNABLE)
# NOTE: Replace these with your actual imported functions for full functionality.
# ======================================================================

# Global constants (from globals.py)
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
time_window = 10        # number of frames to keep in buffer

class Line:
    """A class to hold the properties of a detected lane line."""
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
    """MOCK: Returns dummy camera matrix and distortion coefficients."""
    # In a real app, this would perform cv2.calibrateCamera
    time.sleep(1) # Simulate calibration time
    st.write("Camera calibration simulated.")
    # Return dummy values: ret, mtx, dist, rvecs, tvecs
    return True, np.eye(3), np.zeros((4, 1)), None, None

def undistort(img, mtx, dist, verbose=False):
    """MOCK: Returns the image unchanged."""
    # In a real app, this would perform cv2.undistort
    return img

def binarize(img, verbose=False):
    """MOCK: Returns a grayscale version of the image (simulated binary)."""
    # In a real app, this would perform color and gradient thresholding
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def birdeye(img, verbose=False):
    """MOCK: Returns the image and dummy perspective matrices."""
    # In a real app, this would perform cv2.getPerspectiveTransform and cv2.warpPerspective
    h, w = img.shape[:2]
    M = np.eye(3)
    Minv = np.eye(3)
    # The output image must be 3-channel for blending later
    if len(img.shape) == 2:
        img_birdeye = np.dstack([img] * 3) 
    else:
        img_birdeye = img.copy()
    return img_birdeye, M, Minv

def get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=9, verbose=False):
    """MOCK: Returns dummy line fits and a black image."""
    h, w = img_birdeye.shape[:2]
    
    # Simulate line detection and update
    line_lt.update_line(np.linspace(0, h, 100), np.linspace(w*0.3, w*0.3 + 10, 100), [0, 0, 0], 600)
    line_rt.update_line(np.linspace(0, h, 100), np.linspace(w*0.7, w*0.7 + 10, 100), [0, 0, 0], 700)
    
    # Mock image fit
    img_fit = np.zeros_like(img_birdeye)
    cv2.putText(img_fit, "Sliding Windows (MOCK)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return line_lt, line_rt, img_fit

def get_fits_by_previous_fits(img_birdeye, line_lt, line_rt, verbose=False):
    """MOCK: Returns dummy line fits and a black image."""
    h, w = img_birdeye.shape[:2]

    # Simulate line detection and update
    line_lt.update_line(np.linspace(0, h, 100), np.linspace(w*0.3, w*0.3 + 5, 100), [0, 0, 0], 650)
    line_rt.update_line(np.linspace(0, h, 100), np.linspace(w*0.7, w*0.7 - 5, 100), [0, 0, 0], 750)
    
    # Mock image fit
    img_fit = np.zeros_like(img_birdeye)
    cv2.putText(img_fit, "Previous Fits (MOCK)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return line_lt, line_rt, img_fit

def draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state):
    """MOCK: Returns the image with a colored overlay (simulated)."""
    # In a real app, this would create the lane area polygon and warp it back
    img = img_undistorted.copy()
    h, w = img.shape[:2]
    # Simple green rectangle to simulate the lane overlay
    cv2.rectangle(img, (int(w*0.3), h-50), (int(w*0.7), h), (0, 255, 0), cv2.FILLED)
    return img

# ======================================================================
# CORE LANE DETECTION PIPELINE FUNCTIONS
# ======================================================================

# Global variables initialized once
processed_frames = 0
line_lt = Line(buffer_len=time_window)
line_rt = Line(buffer_len=time_window)
TARGET_WIDTH = 960 # New variable for the speed optimization

def prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter):
    """Adds visual feedback thumbnails and metrics to the final output frame."""
    h, w = blend_on_road.shape[:2]
    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)
    off_x, off_y = 20, 15

    # Create dark header area
    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, (0, 0), (w, thumb_h+2*off_y), (0,0,0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(mask, 0.2, blend_on_road, 0.8, 0)

    # Prepare thumbnails
    thumb_binary = np.dstack([cv2.resize(img_binary, (thumb_w, thumb_h))]*3) * 255
    thumb_birdeye = cv2.resize(img_birdeye, (thumb_w, thumb_h)) # img_birdeye is already 3-channel in mock
    thumb_img_fit = cv2.resize(img_fit, (thumb_w, thumb_h))

    # --- DEFENSIVE CHECK AND PLACEMENT ---
    # Calculate the end position of the third thumbnail (start + width)
    end_x_thumb_3 = 3*off_x + 3*thumb_w

    if end_x_thumb_3 < w:
        # Place thumbnails (Original line indices confirmed to be mathematically sound)
        blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary
        blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*off_x+2*thumb_w, :] = thumb_birdeye
        blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*off_x+3*thumb_w, :] = thumb_img_fit
    else:
        # Fallback message if thumbnails don't fit
        cv2.putText(blend_on_road, "Thumbnails skipped (Frame too narrow)", (off_x, off_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # Add metrics text
    mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, f'Curvature radius: {mean_curvature_meter:.2f}m', (860, 60), font, 0.9, (255,255,255), 2)
    cv2.putText(blend_on_road, f'Offset from center: {offset_meter:.2f}m', (860, 130), font, 0.9, (255,255,255), 2)

    return blend_on_road

def compute_offset_from_center(line_lt, line_rt, frame_width):
    """Calculates the offset of the car from the lane center."""
    # MOCK: Use simplified logic since real line data is complex
    if line_lt.detected and line_rt.detected:
        # Assuming the 'all_x' are the x-coordinates on the bird's eye view
        line_lt_bottom = np.mean(line_lt.all_x[-10:]) if line_lt.all_x is not None else frame_width * 0.3
        line_rt_bottom = np.mean(line_rt.all_x[-10:]) if line_rt.all_x is not None else frame_width * 0.7
        
        lane_center = (line_lt_bottom + line_rt_bottom) / 2
        midpoint = frame_width / 2
        offset_pix = lane_center - midpoint
        offset_meter = xm_per_pix * offset_pix
    else:
        offset_meter = 0.0 # Default to 0.0 if lines are lost
    return offset_meter

def process_pipeline(frame, mtx, dist, keep_state=True):
    """
    The main processing function.
    Optimization: Frame is resized down early to speed up all subsequent operations.
    """
    global line_lt, line_rt, processed_frames
    
    # 1. SPEED UP: Resize frame early if needed (reduces computation for all subsequent steps)
    frame_width_original = frame.shape[1]
    
    if frame_width_original > TARGET_WIDTH:
        scale_factor = TARGET_WIDTH / frame_width_original
        new_dim = (TARGET_WIDTH, int(frame.shape[0] * scale_factor))
        frame_resized = cv2.resize(frame, new_dim, interpolation=cv2.INTER_LINEAR)
    else:
        frame_resized = frame

    # Get the working dimensions
    working_frame_width = frame_resized.shape[1]

    # 2. Start Pipeline with the resized frame
    img_undistorted = undistort(frame_resized, mtx, dist, verbose=False)
    img_binary = binarize(img_undistorted, verbose=False)
    img_birdeye, M, Minv = birdeye(img_binary, verbose=False)

    if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        line_lt, line_rt, img_fit = get_fits_by_previous_fits(img_birdeye, line_lt, line_rt, verbose=False)
    else:
        line_lt, line_rt, img_fit = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=9, verbose=False)

    offset_meter = compute_offset_from_center(line_lt, line_rt, frame_width=working_frame_width)
    
    # 3. Blending
    blend_on_road = draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state)
    blend_output = prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter)
    
    processed_frames += 1
    
    # 4. Optional: Resize back to original size for output video quality (skip for max speed)
    # If we skip this, the VideoWriter must be set to the resized dimensions.
    if frame_width_original > TARGET_WIDTH:
        blend_output = cv2.resize(blend_output, (frame_width_original, frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        
    return blend_output

# ======================================================================
# STREAMLIT UI AND VIDEO PROCESSING LOOP (CORRECTED)
# ======================================================================

st.title("Advanced Lane Detection 🚗 (Optimized)")
st.write("Upload a video or use the sample video to detect lanes in real-time. Processing speed is optimized via frame downscaling.")

# Upload or sample selection
option = st.radio("Select Video Option", ["Upload Video", "Use Sample Video"])

# Note: The actual 'project_video.mp4' needs to be accessible for the sample to work.
SAMPLE_VIDEO_PATH = "project_video.mp4" # Placeholder name

video_file = None
if option == "Upload Video":
    video_file = st.file_uploader("Upload your video (MP4)", type=["mp4"])
elif option == "Use Sample Video":
    st.write(f"Using a mock sample video path: `{SAMPLE_VIDEO_PATH}`. Please ensure this file is available in the real environment.")
    # For a purely self-contained app, we must use the uploaded video only, 
    # but we'll allow the placeholder path for demonstration.
    video_path = SAMPLE_VIDEO_PATH

if video_file or (option == "Use Sample Video" and os.path.exists(SAMPLE_VIDEO_PATH)):
    
    # --- Step 1: Handle Input Path ---
    if option == "Upload Video":
        # Save uploaded video temporarily for OpenCV processing
        tfile = NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        video_path = tfile.name
    
    # --- Step 2: Initialize Calibration and Video Reader ---
    
    # We must ensure this mock file exists for the demonstration structure to work
    if not os.path.exists(video_path):
        st.error(f"Error: Video file not found at path '{video_path}'. Cannot proceed.")
        st.stop()

    with st.spinner("Calibrating Camera... ⏳"):
        ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')
        if not ret:
            st.error("Camera calibration failed or returned dummy data.")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Could not open video file at {video_path}")
        st.stop()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    st.write(f"Video resolution: {frame_width}x{frame_height} @ {fps:.1f} FPS.")
    st.write(f"Processing will be optimized by resizing frames to a maximum width of {TARGET_WIDTH}px.")
    
    
    # --- Step 3: Set up Output Writer (FIXED) ---
    
    # Use mkstemp to get a reliable, clean temporary path
    fd, temp_out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd) # Close file descriptor since cv2.VideoWriter will open it

    # Check if we need to adjust output resolution based on optimization
    if frame_width > TARGET_WIDTH:
        output_width = frame_width # We resize the final frame back up in process_pipeline
        output_height = frame_height
    else:
        output_width = frame_width
        output_height = frame_height

    # Use a widely supported codec ('mp4v' works well with temporary files)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_out_path, fourcc, fps, (output_width, output_height))
    
    # Reset globals for a clean run
    processed_frames = 0
    line_lt = Line(buffer_len=time_window)
    line_rt = Line(buffer_len=time_window)

    # --- Step 4: Processing Loop ---
    status_text = st.empty()
    frame_placeholder = st.empty()
    
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_pipeline(frame, mtx, dist, keep_state=True)
        out.write(processed_frame)
        
        # Display every N frames in Streamlit to improve UI responsiveness
        if frame_count % 5 == 0:
            frame_placeholder.image(processed_frame, channels="BGR", use_column_width=True, caption=f"Frame {frame_count}")
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            status_text.text(f"Processing frame {frame_count}...")


    # --- Step 5: Finalization and Display (FIXED) ---
    
    cap.release()
    out.release()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    status_text.text(f"Processing Complete! Processed {frame_count} frames in {total_time:.2f} seconds.")
    
    try:
        st.video(temp_out_path) # Display the correctly saved final video
    except Exception as e:
        st.error(f"Error displaying final video: {e}")
        st.info("The file was processed, but the browser may not support the temporary MP4 codec.")

    st.success("Processing Complete ✅")
    
    # Cleanup temporary files
    if 'tfile' in locals() and os.path.exists(tfile.name):
        os.remove(tfile.name)
    if os.path.exists(temp_out_path):
        os.remove(temp_out_path)
