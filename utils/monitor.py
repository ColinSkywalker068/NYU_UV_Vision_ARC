import cv2
import config
GREEN_COLOR = (0, 255, 0)
import os
os.environ['GDK_SYNCHRONIZE'] = '1'

#Class to draw bounding boxes and add texts to the image and display it 
class monitor:
    
    # Constructor
    def __init__(self):
        """
        Constructor. Following are the attributes of the class:
        window_name: Name of the window of cv2.imshow 
        """
        self.window_name = "MONITOR"
        
    def draw_on_image(self, image, bboxes, messages):
        """
        Displays the frames after drawing the bounding boxes and thhe text on the input image.
        Parameters: 
            image: Input image on which the annotations are to be drawn
            bboxes: A list of tuples, each tuple containing the 6 specs of a bounding box
            messages: List of strings that are to be printed on the image
        """
        if len(bboxes) == 0:
            cv2.imshow(self.window_name, image)
            return
        # Create a copy of the image
        annotated_image = image.copy()

        # Collect bounding boxes, labels, and confidences
        bbxs = []
        labels = []
        confs = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, conf, label = bbox
            bbxs.append((x_min, y_min, x_max, y_max))
            labels.append(label)
            confs.append(conf)

        # Draw all bounding boxes
        if len(bbxs) > 0:
            annotated_image = self.write_bbx_frame(annotated_image, bbxs, labels, confs)

        # Format and join the messages
        messages = " ".join(messages)

        # Put in the text
        cv2.putText(annotated_image, messages, (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display the image
        cv2.imshow(self.window_name, annotated_image)

    def write_bbx_frame(self, color_image, bbxs, labels, confs):
        """
        Draws a bounding box on the input image
        Parameters:
            color_image: Input image on which the bbox is to be drawn
            bbxs: List of size four indicating x_min, y_min, x_max, y_max of the bounding box to be drawn
            label: Name of class of the object enclosed in the bounding box
            conf: Confidence score of the prediction by the model"""
        # Iterate over bounding boxes
        for bbx, label, conf in zip(bbxs, labels, confs):
            # Convert coordinates to integers
            x_min, y_min, x_max, y_max = map(int, bbx)
            # Display the bounding box
            if config.detection["CLASSIC"]:
                cv2.rectangle(color_image, bbx, GREEN_COLOR, 2)
                cv2.circle(color_image, (x_min+x_max//2,y_min+y_max//2), radius=3, color=(0, 0, 255), thickness=-1)
                
            else:
                cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), GREEN_COLOR, 2)
                cv2.circle(color_image, ((x_min+x_max)//2,(y_min+y_max)//2), radius=3, color=(0, 0, 255), thickness=-1)
            
            # Add label with the confidence
            label_conf = f"{label} {conf:.2f}"
            cv2.putText(color_image, label_conf, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN_COLOR, 2, cv2.LINE_AA)

        return color_image
