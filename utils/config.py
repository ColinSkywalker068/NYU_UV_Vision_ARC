# general specifications
general = {
    'DEBUG_LOGS': True
}

# camera specifications
camera = {
    'DEPTH_CAPTURE': False,
    'CAMERA_INDEX': './vid.mp4',
    'FRAME_SIZE': (640, 480),
    'FRAME_RATE': 30,
    'RECORD': False,
    'HFOV': 69.4,
    'VFOV': 42.5
}


# yolo model specification 
yolo = {
    'REPO_PATH': 'ultralytics/yolov5',
    'MODEL_SOURCE': 'custom',
    'MODEL_WEIGHTS_PATH': './../Algorithm/pt_files/best.pt',
}

# yolo model with tensorrt weights specification
yolo_tensorrt = {
    'REPO_PATH': 'ultralytics/yolov5',
    'MODEL_SOURCE': 'custom',
    'MODEL_WEIGHTS_PATH': './../Algorithm/pt_files/best.engine',
}

# specifications for detection
detection = {
    'DEBUG': True,
    'DISPLAY': True,
    'DETECTION_CONFIDENCE_CUTOFF': 0.25,
    'NO_TO_DETECT': 10,
    'UART': False,
    'IMU': False
}

# drawing specifications
draw = {
    'bbox': {
        'COLOR': (255, 0, 0),
        'THICKNESS': 2
    }
}

# gameplay specifications
# note we could record the area of everything not red, then use bitwise
# not to get the area of everything red, but our method is easier.

gameplay = {
    'ALLY_TEAM_COLOR': 'RED',
    'OPPONENT_TEAM_COLOR': 'BLUE',
    'HSV_LOWER_RED': (0, 20, 228),
    'HSV_UPPER_RED': (18, 255, 254),
    'HSV_LOWER_RED_2': (134, 0, 0),
    'HSV_UPPER_RED_2': (255, 255, 251),
    'HSV_LOWER_BLUE': (60, 30, 131),
    'HSV_UPPER_BLUE': (121, 255, 255)
}
