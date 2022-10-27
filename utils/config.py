# general specifications
general = {
    'DEBUG_LOGS': True
}

# camera specifications
camera = {
    'DEPTH_CAPTURE': True,
    'CAMERA_INDEX': 1,
    'FRAME_SIZE': (640, 360),
    'FRAME_RATE': 60
}


# yolo model specification
yolo = {
    'REPO_PATH': 'ultralytics/yolov5',
    'MODEL_SOURCE': 'custom',
    'MODEL_WEIGHTS_PATH': './../Algorithm/pt_files/best.pt',
}

# specifications for detection
detection = {
    'DEBUG': True,
    'DISPLAY': True,
    'DETECTION_CONFIDENCE_CUTOFF': 0.25,
    'NO_TO_DETECT': 1
}

# drawing specifications
draw = {
    'bbox': {
        'COLOR': (255, 0, 0),
        'THICKNESS': 2
    }
}

# gameplay specifications
gameplay = {
    'ALLY_TEAM_COLOR': 'BLUE',
    'OPPONENT_TEAM_COLOR': 'BLUE',
    'HSV_LOWER_RED': (0, 4, 226),
    'HSV_UPPER_RED': (60, 255, 255),
    'HSV_LOWER_BLUE': (68, 38, 131),
    'HSV_UPPER_BLUE': (113, 255, 255)
}
