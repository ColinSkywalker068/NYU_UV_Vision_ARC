# general specifications
general = {
    "USE_V_DRIVER": False,
    "DEBUG_LOGS": False, 
    "LOG_PATH": "../logs/logs.log"
    }

# camera specifications
camera = {
    "DEPTH_CAPTURE": True,
    "CAMERA_INDEX": 0,
    "FRAME_SIZE": (640, 480),
    "FRAME_RATE": 30,
    "RECORD": False,
    "CameraVersion": "D455-2"
}


# yolo model specification
yolo = {
    "REPO_PATH": "ultralytics/yolov5",
    "MODEL_SOURCE": "custom",
    "MODEL_WEIGHTS_PATH": "./../Algorithm/pt_files/better_yolo.pt",
}

# yolo model with tensorrt weights specification
yolo_tensorrt = {
    "ENABLE": True,
    "REPO_PATH": "ultralytics/yolov5",
    "MODEL_SOURCE": "custom",
    "MODEL_WEIGHTS_PATH": "./../Algorithm/pt_files/better_yolo.engine",
}

# specifications for detection
detection = {
    "DEBUG": False,
    "DISPLAY":False,
    "DETECTION_CONFIDENCE_CUTOFF": 0.5,
    "NO_TO_DETECT": 2,
    "UART": True,
    "IMU": False,
    "TRACKING": True,
    "USE_CUDA_FLAGGER": False,
    "FIDUCIAL": False,
}

# drawing specifications
draw = {"bbox": {"COLOR": (255, 0, 0), "THICKNESS": 2}}

# gameplay specifications
# note we could record the area of everything not red, then use bitwise
# not to get the area of everything red, but our method is easier.

gameplay = {
    "ALLY_TEAM_COLOR": "BLUE",
    "OPPONENT_TEAM_COLOR": "RED",
    "HSV_LOWER_RED": (0, 20, 228),
    "HSV_UPPER_RED": (18, 255, 254),
    "HSV_LOWER_RED_2": (134, 0, 0),
    "HSV_UPPER_RED_2": (255, 255, 251),
    "HSV_LOWER_BLUE": (60, 45, 131),
    "HSV_UPPER_BLUE": (121, 250, 255),
}

fiducials = {
    "camera_intrinsic_matrix": {
        "camera_focal_length_x": 387.909576416016,
        "camera_center_x": 326.653747558594,
        "camera_focal_length_y": 386.904052734375,
        "camera_center_y": 237.997650146484,
    },
    "camera_distortion_coefficients": {
        "camera_dist_coeff_k1": -0.07231168346387425,
        "camera_dist_coeff_k2": 0.054653045623148634,
        "camera_dist_coeff_p1": 0.013779335161555138,
        "camera_dist_coeff_p2": 0.005759852619212033,
        "camera_dist_coeff_k3": 0.15626401335130446,
    },
    "markers": [
        {"marker_id": "A", "marker_file": "./../fiducial_markers/marker_A.png"},
        {"marker_id": "B", "marker_file": "./../fiducial_markers/marker_B.png"},
        {"marker_id": "C", "marker_file": "./../fiducial_markers/marker_C.png"},
        {"marker_id": "D", "marker_file": "./../fiducial_markers/marker_D.png"},
    ],
    "marker_size": {"height": 150, "width": 150, "metric": "mm"},
}


camera_intrinsics = {
    "D435-1": {
        "Width": 640,
        "Height": 480,
        "PPX": 316.750549316406,
        "PPY": 235.639846801758,
        "FX": 608.576538085938,
        "FY": 608.600036621094,
        "DistortionModel": "BrownConrady",
        "Coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "HFOV":  55.47,
        "VFOV": 43.04,
        "H_OFFSET": -0.0022,
        "V_OFFSET": 0.0002,
    },
    "D435-2": {
        "Width": 640,
        "Height": 480,
        "PPX": 321.325225830078,
        "PPY": 246.889373779297,
        "FX": 607.672729492188,
        "FY": 607.150573730469,
        "DistortionModel": "BrownConrady",
        "Coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "HFOV":  55.54,
        "VFOV": 43.13,
        "H_OFFSET": -0.0022,
        "V_OFFSET": 0.0,
    },
    "D455-1": {
        "Width": 640,
        "Height": 480,
        "PPX": 324.895385742188,
        "PPY": 236.558441162109,
        "FX": 386.565704345703,
        "FY": 385.639862060547,
        "DistortionModel": "BrownConrady",
        "Coeffs": [-0.0544622577726841, 0.0625528246164322, -0.000655769952572882, 0.00022078184701968, -0.0200711339712143],
        "HFOV":  79.23,
        "VFOV": 63.79,
        "H_OFFSET": -0.0022,
        "V_OFFSET": 0.0002,
    },
    "D455-2": {
        "Width": 640,
        "Height": 480,
        "PPX": 326.653747558594,
        "PPY": 237.997650146484,
        "FX": 386.750671386719,
        "FY": 385.748138427734,
        "DistortionModel": "BrownConrady",
        "Coeffs": [-0.0544583909213543, 0.0612407959997654, -0.00126419449225068, 0.000465608172817156, -0.0198560617864132],
        "HFOV":  79.2,
        "VFOV": 63.78,
        "H_OFFSET": 0.007487,
        "V_OFFSET": 0.004,
    }
}