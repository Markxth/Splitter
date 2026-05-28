import streamlit as st
from PIL import Image
from time import sleep
import cv2 as cv
import numpy as np 

def split_sort(contours, image):
    if contours is None : 
        return [] 
    limit = image.shape[0] * 0.02  # was 0.2, too large
    bboxes = [cv.boundingRect(c) for c in contours]
    bboxes = sorted(bboxes, key=lambda b: b[1])
    rows = []
    current_row = [bboxes[0]]

    for bbox in bboxes[1:]:
        if abs(bbox[1] - current_row[0][1]) < limit:
            current_row.append(bbox)
        else:
            rows.append(sorted(current_row, key=lambda z: z[0]))
            current_row = [bbox]

    rows.append(sorted(current_row, key=lambda q: q[0]))
    return [box for row in rows for box in row]


def splitter(image):
    file_bytes = np.frombuffer(image.read(), np.uint8)
    bnaimage = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
    bnaimage = cv.GaussianBlur(bnaimage, (5,5), 0) 
    thresh_image = cv.adaptiveThreshold(bnaimage, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)  #to match black on white 
    contours, _ = cv.findContours(thresh_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    image_area = bnaimage.shape[0] * bnaimage.shape[1] 
    valid_contours = []
    for c in contours : 
        area = cv.contourArea(c)
        if( 0.02 * image_area) < area < (0.98 * image_area):
            valid_contours.append(c)
    
    panels_sorted = split_sort(valid_contours, bnaimage)
    panels = []
    for x, y, w, h in panels_sorted:
        panel = bnaimage[y:y+h, x:x+w]  # was w:w+h, should be x:x+w
        panels.append(panel)
    return panels


def main():
    st.title("Image Splitter")
    st.header("Initial Website")
    st.text("Upload an image to be split in different parts")

    file_uploaded = st.file_uploader("Upload a file", type=["png","jpg","jpeg"])
    if file_uploaded is not None:
        st.success("File uploaded successfully!")

        with st.spinner("Splitting..."):
            panels = splitter(file_uploaded)  # was outside if block, panels not captured
            for i, panel in enumerate(panels):
                st.image(panel, caption=f"Image number {i+1}")  # caption= needed as keyword arg
        st.toast("Splitting the image has finished!")


if __name__ == "__main__":
    main()