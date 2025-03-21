import cv2
from crowd_detection import CrowdDetector, DepthEstimator, CrowdAnalyzer, Visualizer

# Load the models
detector = CrowdDetector()
depth_estimator = DepthEstimator()
analyzer = CrowdAnalyzer()
visualizer = Visualizer()

# Load image
image_path = "images/full.jpeg" 
frame = cv2.imread(image_path)

# Detect people
boxes = detector.detect_people(frame)

# Estimate depth
depth_map = depth_estimator.get_depth_map(frame)

# Calculate density
crowd_ratio, density_level = analyzer.calculate_density(boxes, depth_map, frame.shape)

# Draw final overlays
output_frame = visualizer.draw_overlays(frame, boxes, density_level, crowd_ratio)

# Show image locally
cv2.imshow("AHHH_PEOPLE Output", output_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()