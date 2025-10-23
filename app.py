import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time

# ------------------------------
# Lane Detection Functions
# ------------------------------

def find_lane_pixels(image):
    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((image, image, image)) * 255
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 100
    minpix = 50
    window_height = int(image.shape[0] // nwindows)

    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = image.shape[0] - (window + 1) * window_height
        win_y_high = image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 4)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 4)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    return left_fitx, right_fitx, ploty

def search_around_poly(image):
    margin = 100
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx, lefty, rightx, righty, out_img = find_lane_pixels(image)
    if ((len(leftx) == 0) or (len(rightx) == 0) or (len(righty) == 0) or (len(lefty) == 0)):
        out_img = np.dstack((image, image, image)) * 255
        left_curverad = 0
        right_curverad = 0
        left_fitx = None
        right_fitx = None
        ploty = None
    else:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                          (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                           (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fitx, right_fitx, ploty = fit_poly(image.shape, leftx, lefty, rightx, righty)

        ym_per_pix = 30 / 720
        xm_per_pix = 3.7 / 650

        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        out_img = np.dstack((image, image, image)) * 255
        window_img = np.zeros_like(out_img)
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        points = np.hstack((left, right))
        out_img = cv2.fillPoly(out_img, np.int_(points), (0, 200, 255))

    return out_img, left_curverad, right_curverad, left_fitx, right_fitx, ploty

# ------------------------------
# Preprocessing Functions
# ------------------------------

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 50, 150)
    cropped = region_of_interest(canny)
    return cropped

def calculate_steering_angle(image, left_fitx, right_fitx, ploty):
    if left_fitx is None or right_fitx is None or ploty is None:
        return 0
    left_bottom = left_fitx[-1]
    right_bottom = right_fitx[-1]
    lane_center = (left_bottom + right_bottom) / 2
    image_center = image.shape[1] / 2
    offset = lane_center - image_center
    max_offset = 200
    angle = (offset / max_offset) * 30
    return angle

def create_steering_wheel(angle):
    wheel_size = 200
    wheel = np.zeros((wheel_size, wheel_size, 3), dtype=np.uint8)
    center = (wheel_size // 2, wheel_size // 2)
    radius = wheel_size // 2 - 10

    cv2.circle(wheel, center, radius, (50, 50, 50), -1)
    cv2.circle(wheel, center, radius, (200, 200, 200), 3)
    cv2.circle(wheel, center, radius // 2, (100, 100, 100), -1)
    cv2.circle(wheel, center, radius // 2, (200, 200, 200), 2)

    for i in range(4):
        spoke_angle = i * 90 + angle
        rad = np.deg2rad(spoke_angle)
        x = int(center[0] + radius * 0.8 * np.cos(rad))
        y = int(center[1] + radius * 0.8 * np.sin(rad))
        cv2.line(wheel, center, (x, y), (200, 200, 200), 4)

    cv2.circle(wheel, center, radius // 4, (50, 50, 50), -1)
    cv2.circle(wheel, center, radius // 4, (200, 200, 200), 2)
    return wheel

# ------------------------------
# Streamlit App
# ------------------------------

st.title("Advanced Lane Detection with Polynomial Fitting")
st.write("This app uses sliding window approach and polynomial fitting for lane detection.")

# Sidebar Controls
st.sidebar.title("Controls")
video_source = st.sidebar.selectbox("Video Source", ["Upload Video", "Use Sample Video"])
show_binary = st.sidebar.checkbox("Show Binary Image", False)
show_windows = st.sidebar.checkbox("Show Sliding Windows", True)
stop_video = st.sidebar.button("Stop Video")

# Video Source
if video_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = io.BytesIO(uploaded_file.read())
        with open("temp_video.mp4", "wb") as out:
            out.write(tfile.getbuffer())
        vid = cv2.VideoCapture("temp_video.mp4")
    else:
        st.sidebar.warning("Please upload a video file")
        vid = None
else:
    vid = cv2.VideoCapture("test2.mp4")

# Layout Columns
col1, col2, col3 = st.columns([2, 2, 1])

# ------------------------------
# Video Processing Loop
# ------------------------------
if vid is not None:
    video_placeholder = col1.empty()
    binary_placeholder = col2.empty() if show_binary else None
    result_placeholder = col2.empty() if not show_binary else col3.empty()
    steering_placeholder = col3.empty() if not show_binary else col2.empty()
    angle_placeholder = st.empty()
    curvature_placeholder = st.empty()

    while vid.isOpened():
        if stop_video:
            break

        ret, frame = vid.read()
        if not ret:
            break

        binary_image = preprocess_frame(frame)

        if show_windows:
            leftx, lefty, rightx, righty, window_img = find_lane_pixels(binary_image)
            window_img_rgb = cv2.cvtColor(window_img, cv2.COLOR_BGR2RGB)
        else:
            window_img_rgb = None

        lane_img, left_curverad, right_curverad, left_fitx, right_fitx, ploty = search_around_poly(binary_image)
        angle = calculate_steering_angle(frame, left_fitx, right_fitx, ploty)
        steering_wheel = create_steering_wheel(angle)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lane_img_rgb = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)
        steering_wheel_rgb = cv2.cvtColor(steering_wheel, cv2.COLOR_BGR2RGB)

        video_placeholder.image(frame_rgb, channels="RGB", caption="Original Video")
        if show_binary:
            binary_img_rgb = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
            binary_placeholder.image(binary_img_rgb, channels="RGB", caption="Binary Image")
            result_placeholder.image(lane_img_rgb, channels="RGB", caption="Lane Detection")
        else:
            result_placeholder.image(lane_img_rgb, channels="RGB", caption="Lane Detection")

        if show_windows and window_img_rgb is not None:
            st.image(window_img_rgb, channels="RGB", caption="Sliding Windows")

        steering_placeholder.image(steering_wheel_rgb, channels="RGB", caption="Steering Wheel")

        avg_curvature = (left_curverad + right_curverad) / 2 if left_curverad > 0 and right_curverad > 0 else 0
        angle_placeholder.write(f"**Steering Angle:** {angle:.1f}°")
        curvature_placeholder.write(f"**Lane Curvature:** {avg_curvature:.0f} m")

        time.sleep(0.03)

    vid.release()
else:
    st.info("Please upload a video or use the sample video.")
