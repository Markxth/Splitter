1. What this is

This is an image splitter built to take an image, split it, and return the results. It is specifically optimised for storyboard images which have inner smaller panels, and so, usually a border. This tool is
optimised to detect those borders, and subtract the panel.

2. How it works

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

The following images may return weird results or just not work, however, see examples/negative for applied examples: 

-random pictures without any sort of structure to them 

3. Current Detection Heuristics

The system currently uses:

Gaussian blur for noise reduction
Adaptive thresholding for robust edge separation under varying lighting conditions
Contour-based segmentation
Relative area filtering heuristics
Bounding-box spatial sorting

This makes the prototype lightweight, fast, and easy to experiment with without requiring machine learning models.

4. Technical Notes

Some important implementation details:

Images are processed in grayscale for simpler contour extraction
Adaptive thresholding is used instead of fixed thresholding to improve robustness across different image conditions
Bounding rectangles are used to crop detected panels
Panels are sorted row-by-row to preserve reading order as much as possible
