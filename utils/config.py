# general specifications
general = {
    "DEBUG_LOGS": False,
    "LOG_PATH": "../logs/logs.log"}

# camera specifications
camera = {
    "DEPTH_CAPTURE": True,
    "CAMERA_INDEX": 0,
    "FRAME_SIZE": (640, 480),
    "FRAME_RATE": 30,
    "RECORD": False,
    "HFOV": 69.4,
    "VFOV": 42.5,
}


# yolo model specification
yolo = {
    "REPO_PATH": "ultralytics/yolov5",
    "MODEL_SOURCE": "custom",
    "MODEL_WEIGHTS_PATH": "./../Algorithm/pt_files/best.pt",
}

# yolo model with tensorrt weights specification
yolo_tensorrt = {
    "ENABLE": False,
    "REPO_PATH": "ultralytics/yolov5",
    "MODEL_SOURCE": "custom",
    "MODEL_WEIGHTS_PATH": "./../Algorithm/pt_files/best.engine",
}

# specifications for detection
detection = {
    "DEBUG": False,
    "DISPLAY": False,
    "DETECTION_CONFIDENCE_CUTOFF": 0.25,
    "NO_TO_DETECT": 10,
    "UART": True,
    "IMU": False,
    "TRACKING": False,
    "USE_CUDA_FLAGGER": False,
}

# drawing specifications
draw = {"bbox": {"COLOR": (255, 0, 0), "THICKNESS": 2}}

# gameplay specifications
# note we could record the area of everything not red, then use bitwise
# not to get the area of everything red, but our method is easier.

gameplay = {
    "ALLY_TEAM_COLOR": "RED",
    "OPPONENT_TEAM_COLOR": "BLUE",
    "HSV_LOWER_RED": (0, 20, 228),
    "HSV_UPPER_RED": (18, 255, 254),
    "HSV_LOWER_RED_2": (134, 0, 0),
    "HSV_UPPER_RED_2": (255, 255, 251),
    "HSV_LOWER_BLUE": (60, 30, 131),
    "HSV_UPPER_BLUE": (121, 255, 255),
}

fiducials = {

    'camera_intrinsic_matrix' : {    
        'camera_focal_length_x' : 1427.9641708247764,
        'camera_center_x' :  974.409732316623,
        'camera_focal_length_y' : 1434.9326534580543,
        'camera_center_y' :  571.1514876178345
    },

    'camera_distortion_coefficients' :{
        'camera_dist_coeff_k1' : -0.07231168346387425,
        'camera_dist_coeff_k2' : 0.054653045623148634,
        'camera_dist_coeff_p1' : 0.013779335161555138,
        'camera_dist_coeff_p2' : 0.005759852619212033,
        'camera_dist_coeff_k3' : 0.15626401335130446
    },


    'markers' :[
        {
            'marker_id' : 'A',
            'marker_file' : './../fiducial_markers/marker_A.png'
        },
        {
            'marker_id' : 'B',
            'marker_file' : './../fiducial_markers/marker_B.png'
        },
            {
            'marker_id' : 'C',
            'marker_file' : './../fiducial_markers/marker_C.png'
        },
            {
            'marker_id' : 'D',
            'marker_file' : './../fiducial_markers/marker_D.png'
        }
    ],

    'marker_size' : {
        'height' : 150,
        'width' : 150,
        'metric' : 'mm'
    }

}
