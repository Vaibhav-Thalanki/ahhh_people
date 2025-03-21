import cv2

class Visualizer:
    def draw_overlays(self, frame, boxes, density_level, crowd_ratio):
        # Draw enclosing rectangle
        if len(boxes) > 0:
            x1s, y1s, x2s, y2s = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            min_x, min_y = int(min(x1s)), int(min(y1s))
            max_x, max_y = int(max(x2s)), int(max(y2s))
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

        # Draw labels
        cv2.rectangle(frame, (0, 0), (450, 180), (0, 0, 0), -1)
        cv2.putText(frame, f'Density Level: {density_level}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Crowd Ratio: {crowd_ratio:.4f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Highlight alert
        if density_level in ["High", "Very High"]:
            cv2.putText(frame, "⚠️ ALERT: High Crowd Density", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame
