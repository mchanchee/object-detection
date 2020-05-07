from imageai.Detection import ObjectDetection

import cv2
import os

if __name__ == "__main__":

    # Get the model weights
    execution_path = os.getcwd()
    model_path = os.path.join(execution_path, "yolo-tiny.h5")

    # Set up the detector
    detector = ObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel(detection_speed="flash")

    # Set up the webcam
    camera = cv2.VideoCapture(0)

    while cv2.waitKey(1) != 13: # 13 is the Enter key
        
		# detect objects
        ret, frame = camera.read()

        detected_image_array, detections = detector.detectObjectsFromImage(input_type="array", 
                                                                           input_image=frame, 
                                                                           output_type="array", 
                                                                           minimum_percentage_probability=20, 
                                                                           display_percentage_probability=True)

        for detection in detections:
            print(detection["name"] , " : ", detection["percentage_probability"], " : ", detection["box_points"] )

        cv2.imshow("Object detection", detected_image_array)
        print("--------------------------------")

	# clean up
    camera.release()
    cv2.destroyAllWindows()
       