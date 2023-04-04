import cv2
import fiducial_detection

def main():

    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    fid_detector = fiducial_detection.Fiducial_detector()


    while(ret):
        fid_detector.detectFiducial(frame)
        ret, frame = cap.read()


if __name__=='__main__':
    main()