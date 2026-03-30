{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 .SFNS-Regular_wdth_opsz110000_GRAD_wght2580000;}
{\colortbl;\red255\green255\blue255;\red236\green244\blue251;\red1\green5\blue10;}
{\*\expandedcolortbl;;\cssrgb\c94118\c96471\c98824;\cssrgb\c392\c1569\c3529;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\b\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
import streamlit as st\
import cv2\
import numpy as np\
from PIL import Image\
\
st.set_page_config(page_title="Lung CT Analyzer", layout="wide")\
\
st.title("\uc0\u55358 \u57025  Lung CT Analyzer")\
st.markdown("\uc0\u3623 \u3636 \u3648 \u3588 \u3619 \u3634 \u3632 \u3627 \u3660 \u3616 \u3634 \u3614  CT \u3611 \u3629 \u3604 ")\
\
uploaded = st.file_uploader("\uc0\u3629 \u3633 \u3611 \u3650 \u3627 \u3621 \u3604 \u3616 \u3634 \u3614  CT \u3611 \u3629 \u3604 ", type=['jpg', 'jpeg', 'png', 'bmp'])\
\
if uploaded:\
    image = Image.open(uploaded).convert('L')\
    img_array = np.array(image)\
    \
    st.sidebar.header("\uc0\u9881 \u65039  \u3585 \u3634 \u3619 \u3605 \u3633 \u3657 \u3591 \u3588 \u3656 \u3634 ")\
    window_center = st.sidebar.slider("Window Center", -1000, 1000, -600, 50)\
    window_width = st.sidebar.slider("Window Width", 1, 2000, 1500, 100)\
    contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0, 0.1)\
    threshold = st.sidebar.slider("Threshold", 0, 255, 127, 1)\
    kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 5, 2)\
    \
    # Windowing\
    min_val = window_center - window_width // 2\
    max_val = window_center + window_width // 2\
    windowed = np.clip(img_array, min_val, max_val)\
    windowed = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)\
    \
    # Enhancement\
    enhanced = cv2.convertScaleAbs(windowed, alpha=contrast, beta=0)\
    \
    # Thresholding\
    _, binary = cv2.threshold(enhanced, threshold, 255, cv2.THRESH_BINARY_INV)\
    \
    # Morphology\
    kernel = np.ones((kernel_size, kernel_size), np.uint8)\
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)\
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)\
    \
    # Find contours\
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\
    \
    # Result image\
    result_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)\
    \
    if contours:\
        biggest = max(contours, key=cv2.contourArea)\
        cv2.drawContours(result_img, [biggest], -1, (0, 255, 0), 2)\
        \
        area = cv2.contourArea(biggest)\
        diameter = np.sqrt(4 * area / np.pi) * 0.5\
        \
        st.success(f"\uc0\u3614 \u3610 \u3585 \u3657 \u3629 \u3609 \u3586 \u3609 \u3634 \u3604  ~\{diameter:.2f\} mm")\
    else:\
        st.warning("\uc0\u3652 \u3617 \u3656 \u3614 \u3610 \u3585 \u3657 \u3629 \u3609 ")\
    \
    # Display\
    col1, col2, col3 = st.columns(3)\
    col1.image(image, caption="Original")\
    col2.image(mask, caption="Mask")\
    col3.image(result_img, caption="Result")}
