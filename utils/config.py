# general specifications
general = {
    'DEBUG_LOGS' : True
}

# camera specifications
camera = {
    'DEPTH_CAPTURE' : False,
    'CAMERA_INDEX' : "./../../../Midterm/RMPlateDetection/vid.mp4",
    'FRAME_SIZE' : (640,360),
    'FRAME_RATE' : 30
}


# yolo model specification
yolo = {
    'REPO_PATH' : 'ultralytics/yolov5',
    'MODEL_SOURCE' : 'custom',
    'MODEL_WEIGHTS_PATH': './../Algorithm/pt_files/best.pt',
}

# specifications for detection
detection = {
    'DEBUG' : False,
    'DISPLAY' : True,
    'DETECTION_CONFIDENCE_CUTOFF' : 0.25,
    'NO_TO_DETECT' : 1
}
     
# drawing specifications
draw = {
    'bbox' : {
        'COLOR' : (255,0,0),
        'THICKNESS' : 2
    }
}

# gameplay specifications
gameplay = {
    'ALLY_TEAM_COLOR': 'BLUE',
    'OPPONENT_TEAM_COLOR' : 'RED',
}