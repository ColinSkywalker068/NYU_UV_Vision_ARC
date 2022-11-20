import cv2
import torch
import os
import flagger
import uart_server
import torch.backends.cudnn as cudnn


#  Autoselect kernel with best performance on current hardware
cudnn.benchmark = True


class plate_detector:

    # Constructor
    def __init__(self, repo='ultralytics/yolov5', model='custom', model_file_path='./Algorithm/pt_files/best.pt', tracking=True, flagger=None, uart=False):
        """
        Constructor. Initializes attributes. Uses torch.hub.load() of PyTorch module to load a pytorch model.
        Parameters:
            repo:  url of the repository that has the load scripts for the model
            model: type of model 
            model_file_path: path to the weights of the model
            tracking: True if tracking is to be done else False
            flagger: flagger object (Default: None)
        Attributes:
            model: Object detection model object
            tracker: tracker object
            tracking: True if tracking is enabled else False
            tracker_initialized: True if tracker is already initialized else False
        """

        # Create object detection model
        # or yolov5m, yolov5l, yolov5x, custom
        self.model = torch.hub.load(
            repo, 'custom', path=model_file_path).eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # Deactivate Dropout layers
        self.model.eval()  # set model to evaluation mode

        # @TODO: convert to tensorrt

        # Check if tracking is required
        if tracking:
            # Create the tracker object
            self.tracking = True
            self.tracker = cv2.legacy.TrackerMedianFlow_create()
            self.tracker_initialized = False
        else:
            self.tracking = False

        if flagger is not None:
            self.flagger = flagger

        if uart:
            # create a uart object to send/receive data
            self.uart = uart_server.Uart()
        else:
            self.uart = None

    # Process a color frame
    def process_frame(self, color_image, conf_thres=0.25, k=1, debug=False):
        """
        Performs object detection on the input image. 
        Filters out predictions.
        Performs tracking.
        Paramters:
        color_image = input image on which the object detection is to be performed.
        conf_thresh = The threshold value for the confidence score of the predictions by the model. The predicitons below the conf_thresh are discarded.
        k = No. of objects to be outputed from the object detection predictions
        debug = Prints debug logs if True

        Returns:
            bboxes: A list of tuples, each tuple containing the 6 specs of each  bounding box that is predicted
            messages: List of strings that are to be printed on the image
        """

        # Get bounding boxes
        results = self.model(color_image)

        # Post process bounding boxes
        #detections_rows = results.pandas().xyxy

        # use numpy (faster)
        # x_min, y_min, x_max, y_max, conf, class (0 for plate)
        # no cls (only in pandas)

        # detections_rows = [results.xyxy[i].detach().cpu().numpy()
        # for i in range(len(results.xyxy))]

        # 4th column is confidence; for every row in the results (xyxy) that we
        # copy to main memory
        top_k_detections = [row[row[:, 4] >= conf_thres] for row in [
            result.detach().cpu().numpy() for result in results.xyxy]][0][:k]
        # use [0] bc tensor, [:k] bc top k

        bboxes = []
        messages = []

        for i in range(len(top_k_detections)):
            # Get the bounding box of the first object (most confident)
            x_min, y_min, x_max, y_max, conf, label = top_k_detections[i]

            # check if plate is enemy plate
            if label == 0 and self.flagger is not None:
                # (y_min, y_max, x_min, x_max)
                if not self.flagger.check_enemy(color_image[int(y_min):int(y_max), int(x_min):int(x_max)]):
                    continue

            if self.tracking:

                if not self.tracker_initialized:
                    # Initialize the tracker
                    tracker_success = self.tracker.init(
                        color_image, (x_min, y_min, x_max, y_max))
                    self.tracker_initialized = True
                else:

                    # Update tracker
                    tracker_success, bbox = self.tracker.update(color_image)
                    # bboxes.append(bbox)

                    # Add bounding boxes to be drawn on the image
                    if tracker_success:
                        bboxes.append(
                            [x_min, y_min, x_max, y_max, conf, 'plate'])
                        messages.append("Tracking Success")
                    else:
                        print('Tracking Failure')
                        messages.append('Tracking failed for {}'.format(i))

            else:
                # Add bounding boxes to be drawn on the image
                bbox = [x_min, y_min, x_max, y_max, conf, label]
                bboxes.append(bbox)
                messages.append("")

            if self.uart is not None:
                # @TODO: send data to uart server
                self.uart.send_data("Plate detected")

            if debug:
                print("Object {} detected at ({},{}) \n\n\n ({},{})".format(
                    i, x_min, y_min, x_max, y_max))
                # print for windows
                if os.name == 'nt':
                    os.system('cls')
                else:
                    os.system('clear')

        return bboxes, messages
