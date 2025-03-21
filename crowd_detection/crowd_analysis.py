import numpy as np

class CrowdAnalyzer:
    def __init__(self):
        self.density_levels = {
            'Low': (0, 0.01),
            'Medium': (0.01, 0.05),
            'High': (0.05, 0.1),
            'Very High': (0.1, float('inf'))
        }

    def calculate_density(self, boxes, depth_map, frame_shape):
        if len(boxes) == 0:
            return 0, "No People"

        # Find the enclosing crowd box
        x1s, y1s, x2s, y2s = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        min_x, min_y = int(np.min(x1s)), int(np.min(y1s))
        max_x, max_y = int(np.max(x2s)), int(np.max(y2s))
        crowd_box_area = (max_x - min_x) * (max_y - min_y)

        # Compute depth-weighted crowd area
        depth_weighted_area, total_weight = 0, 0
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            person_depth = depth_map[int((y1 + y2) / 2), int((x1 + x2) / 2)]
            depth_weight = max(1 - person_depth, 0.2)  # Closer people count more

            person_area = (x2 - x1) * (y2 - y1) * depth_weight
            depth_weighted_area += person_area
            total_weight += depth_weight

        # Calculate final crowd ratio
        crowd_ratio = (depth_weighted_area / total_weight) / crowd_box_area
        density_level = next(k for k, (lo, hi) in self.density_levels.items() if lo <= crowd_ratio < hi)

        return crowd_ratio, density_level
