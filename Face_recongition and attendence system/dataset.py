import cv2
import os

# Define a function to capture images
def capture_images(person_name, num_images=100, save_path='dataset'):
    # Start video capture
    cap = cv2.VideoCapture(0)
    count = 0

    # Create the directory for the person if it doesn't exist
    person_path = os.path.join(save_path, person_name)
    os.makedirs(person_path, exist_ok=True)

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        # Show the frame
        cv2.imshow('Capture Images - Press Q to Quit', frame)

        # Save the image
        img_name = os.path.join(person_path, f"{person_name}_{count}.jpg")
        cv2.imwrite(img_name, frame)
        count += 1

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Capture images for a new person
person_name = input("Enter the person's name: ")
capture_images(person_name)
