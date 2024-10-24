import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the HSV range for blue laser
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        distances = []  # List to store distances

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 100:  # Adjustable threshold based on actual conditions
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

                # Get the bounding box of the contour
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate the brightness of the contour
                mask_rect = mask[y:y+h, x:x+w]
                brightness = cv2.mean(frame[y:y+h, x:x+w], mask=mask_rect)[:3]

                # Print brightness value for debugging(represents the average color intensity of the detected region in the BGR (Blue, Green, Red) color space)
                print(f"Brightness: {brightness}")

                # Only estimate distance if brightness is sufficiently high
                if brightness[0] > 50:  #I'm using low light
                    estimated_distance = (2.0 / (area ** 0.5)) * 100  # Adjust based on actual conditions, I'm conducting formula validation in a relatively dark environment.
                    distances.append(estimated_distance)

                    # Print estimated distance for debugging
                    print(f"Estimated Distance: {estimated_distance:.1f} cm")

                    if estimated_distance < 200:
                        cv2.putText(frame, f'Distance: {estimated_distance:.1f} cm', (cX, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate average distance
        if distances:
            average_distance = sum(distances) / len(distances)
            cv2.putText(frame, f'Avg Distance: {average_distance:.1f} cm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('Blue Laser Detection', frame)
        # Exit
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
