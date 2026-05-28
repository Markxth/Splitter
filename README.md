# Splitter Documentation

## 1. What this is

This is an image splitter built to take an image, split it, and return the results. It is specifically optimised for storyboard images which have inner smaller panels, and so, usually a border. This tool is
optimised to detect those borders, and subtract the panel.

## 2. How it works

The uploaded image is first read as raw bytes and decoded into a grayscale OpenCV image.
The system then applies several preprocessing and filtering stages in order to isolate likely storyboard panels.

The current processing pipeline is:

Read uploaded image bytes into a NumPy array
Decode the image in grayscale using OpenCV
Apply Gaussian Blur to reduce noise and smooth edges
Apply Adaptive Gaussian Thresholding to convert the image into a high-contrast binary image
Detect contours using OpenCV contour extraction
Compute the total image area
Filter contours based on relative area thresholds in order to remove:
extremely small regions/noise
contours covering almost the entire image
Sort detected panels spatially (top-to-bottom, left-to-right)
Crop each detected panel from the original image
Return the resulting list/vector of split panels

The contour filtering currently assumes storyboard-like structures with relatively clear panel boundaries and spacing. Whilst it may work on other pictures, it was built with the aforementioned prototype
structure in mind.

## 3. Current Detection Heuristics

The system currently uses:

Gaussian blur for noise reduction
Adaptive thresholding for robust edge separation under varying lighting conditions
Contour-based segmentation
Relative area filtering heuristics
Bounding-box spatial sorting

This makes the prototype lightweight, fast, and easy to experiment with without requiring machine learning models.

## 4. Technical Notes

Some important implementation details:

Images are processed in grayscale for simpler contour extraction
Adaptive thresholding is used instead of fixed thresholding to improve robustness across different image conditions
Bounding rectangles are used to crop detected panels
Panels are sorted row-by-row to preserve reading order as much as possible

## 5. Use Cases

Since this tool wss created for storyboard images, a storyboard image will work perfectly, provided it does not have multiple storyboards within a storyboard - see examples/negative/example1. For an example of a good use case, see examples/positive/example1

The following use cases may return undesirable results and/or unreliable results. Since this tool was built with storyboard images in mind, it is natural for this to happen, and so the following use cases will most likely NOT work or work in an undesirable manner : 

   - Images with a lot of small pannels - see examples/negative/example1
   - Random images with a given object, animal, person, etc. For example : an image with a kangaroo will not work, specifically one will received an "index out of range" error -  see examples/negative/example2
     

## 6. Extra notes/Others

If one changes the name of the .py file for any given reason, and the code does not work anymore, it is due to the way streamlit's cache works. Just run the following line and it will work : 

  taskkill /F /IM python.exe


