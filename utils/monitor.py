import cv2


#Class to draw bounding boxes and add texts to the image and display it 
class monitor:
    
    # Constructor
    def __init__(self):
        self.window_name = "MONITOR"
        
    def draw_on_image(self, image, bboxes, messages):
        
        # Draw all bounding boxes on the image
        for bbox in bboxes:
            conf, label = bbox[-2:]
            bbox = bbox[:4]
            image = self.write_bbx_frame(image, bbox, label, conf)
    
        # Put in the text
        cv2.putText(image, " ".join(messages), (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display the image
        cv2.imshow(self.window_name, image)
             
             
    def write_bbx_frame(self, color_image, bbxs, label, conf):
        # Display the bounding box
        x_min, y_min, x_max, y_max = bbxs
        cv2.rectangle(color_image, (int(x_min), int(y_min)), (int(
            x_max), int(y_max)), (0, 255, 0), 2)  # Draw with green color

        # Add label with the confidence
        label_conf =str(label) + " " + str(conf)
        cv2.putText(color_image, label_conf, (int(x_min), int(
            y_min)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return color_image