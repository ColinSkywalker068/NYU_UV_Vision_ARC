general = {
    'DEBUG_LOGS' : True
}

camera = {
    'DEPTH_CAPTURE' : False,
    'CAMERA_INDEX' : "./../../../Midterm/RMPlateDetection/vid.mp4",
    'FRAME_SIZE' : (640,360),
    'FRAME_RATE' : 30
}

yolo = {
    'REPO_PATH' : 'ultralytics/yolov5',
    'MODEL_SOURCE' : 'custom',
    'MODEL_WEIGHTS_PATH': './../Algorithm/pt_files/best.pt',
}

detection = {
    'DEBUG' : False,
    'DISPLAY' : True,
    'DETECTION_CONFIDENCE_CUTOFF' : 0.25,
    'NO_TO_DETECT' : 1
}
       
draw = {
    'bbox' : {
        'COLOR' : (255,0,0),
        'THICKNESS' : 2
    }
}