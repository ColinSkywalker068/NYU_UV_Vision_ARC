import cv2
import fiducial_detection
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

def main():

    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    fid_detector = fiducial_detection.Fiducial_detector()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(vertical_axis='y')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(left=-500, right=500)
    ax.set_ylim3d(bottom=-500, top=500)
    ax.set_zlim3d(bottom=-500, top=500)

    # dummy real world for testing
    def plot_real_world(ax):
        # ax.plot3D([0,106,106,0,0],[0,0,150,150,0],[0,106,106,0,0],c='black')
        # ax.plot3D([0,53,106],[0,150,0],[0,53,106],c='black')

        ax.plot3D(xs=[0,150,150,0,0],ys=[0,0,150,150,0],zs=[0,0,0,0,0],c='black')
        #ax.plot3D(xs=[0,0,0],ys=[0,150,0],zs=[0,75,150],c='black')
        


    while(ret):
        ret_dict = fid_detector.detectFiducial(frame)
        plt.pause(0.05)
        ax.clear()
        plot_real_world(ax)
        


        if ret_dict['success']:
            loc = ret_dict['rw_loc_cam'][:3].reshape(3,1)
            camera_plane_rw = ret_dict['camera_plane_rw']
            #print(camera_plane_rw)

            camera_plane_rw = np.append(camera_plane_rw, camera_plane_rw[:1,:],axis=0)
            print(camera_plane_rw)
            #ax.scatter(xs=loc[0], ys=loc[1], zs=loc[2], c='red', s=2)
            ax.plot3D(xs=camera_plane_rw[:,0], ys=camera_plane_rw[:,1], zs=camera_plane_rw[:,2], c='red')

        ret, frame = cap.read()

    plt.show()



if __name__=='__main__':
    main()