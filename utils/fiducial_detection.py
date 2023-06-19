import cv2
import math
import numpy as np
import flagger 
import config
import fid_locs
from math import sin, cos

#Detection of fiducial tutorial is present here: https://colab.research.google.com/drive/1Ic91IODIZCoPxRoc4GIr7LsZ5K0lVVv_#scrollTo=tIlGwYb2V5W1


# Takes in a countour (cnt) of 4 points and reorders them such that they form a rectangle 
# Top-left point first and other points ordered clock-wise
def order_contour(cnt):
  # Calculates center point of contour through average of 4 points
  cx = (cnt[0][0][0] + cnt[1][0][0] + cnt[2][0][0] + cnt[3][0][0])/4.0
  cy = (cnt[0][0][1] + cnt[1][0][1] + cnt[2][0][1] + cnt[3][0][1])/4.0

  # Orders the contour
  if(cnt[0][0][0] <= cx and cnt[0][0][1]<=cy):
    temp = cnt[3].copy()
    cnt[3] = cnt[1]
    cnt[1] = temp
  else:
    temp = cnt[0].copy()
    cnt[0] = cnt[1]
    cnt[1] = temp

    temp2 = cnt[3].copy()
    cnt[3] = cnt[2]
    cnt[2] = temp2

  return cnt



def orientWRTaxes(coords):
  """
  reorders the list of coordinates of a quadrilateral in the order [p1, p2, p3, p4] as shown below
  expect coords to contain 4 points (x,y,z) representing a square
  p2------p3
  |        |
  |        |
  |        |
  p1------p4
  """
  p1 = coords[0]
  p2 = coords[1]
  p3 = coords[2]
  p4 = coords[3]

  # Accept two 3D vectors

  v1 = p3-p2
  v2 = p1-p2

  # Check if the two vectors are orthogonal
  # if np.dot(v1, v2) != 0:
  #     print("The two vectors are not orthogonal.",np.dot(v1,v2))

  # Compute an orthogonal vector (o) to both input vectors
  o = np.cross(v2, v1)

  # Normalize the vector o
  n = o / np.linalg.norm(o)

  # Compute the angle between n and [0,0,1], [0,1,0], and [1,0,0] in degrees
  angle1 = math.degrees(math.acos(np.dot(n, np.array([1,0,0]))))
  angle2 = math.degrees(math.acos(np.dot(n, np.array([0,1,0]))))
  angle3 = math.degrees(math.acos(np.dot(n, np.array([0,0,1]))))

  res = [angle1,angle2,angle3]
  return res, n



def get_contour_bits(im,cnt,bits):
  """ 
  bits = total number of bits in the fiducial. For RM it is 49 bits in total arranged in a 7x7 grid
  img = camera feed frame that contains the fiducial
  cnt = contour of the detected fiducial
  This function takes in the image, extracts the detected fiducial, coverts it into a rectangular image
  and gives out the signature of the detected fiducial
  """
  # Calculates pixel length
  pixelLen = int(math.sqrt(bits))

  # Corners of square / fiducial marker
  corners = np.array([[0,0],[bits,0],[bits,bits],[0,bits]],np.float32)

  # Transformation Matrix that maps the contour points to corners
  M = cv2.getPerspectiveTransform(cnt, corners)

  # Apply transformation onto input image and store in variable
  dst = cv2.warpPerspective(im, M, (bits, bits),flags=cv2.INTER_LINEAR)
  #cv2.imshow(dst) 

  # Converst dst to greyscale
  gs =  cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
  ret, im_thres = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  #cv2.imshow(im_thres)

  # Create rectangular structuring object
  element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

  # Applies erosion element onto im_thres
  im_er = cv2.erode(im_thres,element)
  #cv2.imshow(im_er)

  # Computes szie of im_er and size of each grid cell
  im_size=im_er.shape
  #print("Image Shape:",im_size)
  grid_cell_size = (im_size[0]//pixelLen, im_size[1]//pixelLen)
  #print("Grid Cell size:",grid_cell_size)

  # Loop and calculate mean intensity of each cell
  res = []
  for r in range(pixelLen):
    for c in range(pixelLen):
      y_1 = int(r * pixelLen )
      x_1 = int(c * pixelLen )
      y_2 = y_1 + grid_cell_size[1]
      x_2 = x_1 + grid_cell_size[0]
      cell = im_er[y_1:y_2,x_1:x_2]
      #cv2_imshow(cell)
      cell_val = cell.mean()/255

      if cell_val>0.3:
        res.append(1)
      else:
        res.append(0)
  
  return res, dst

def load_marker_dictionary(marker,bits):
  
  # Variables to store binary signatature, width, and height of image
  sigs = []
  w = len(marker[0])
  h= len(marker)

  # im_marker = cv2.cvtColor(marker,cv2.COLOR_BGR2GRAY)
  # Creates copy and defines corner coordinates of countour
  im_marker = marker.copy()
  cnt = np.array([[0,0],[w,0],[w,h],[0,h]],np.float32)

  # Loops through each quandrant and calls "get_contour bits" function 
  for i in range(4):
    sig,_ = get_contour_bits(im_marker,cnt,bits)

    # Adds binary signature to list
    sigs.append((sig,i))

    # Rotates image 90°
    im_marker = cv2.rotate(im_marker, cv2.ROTATE_90_CLOCKWISE)

  return sigs

def draw_t_offset_line(rvec, tvec, camera_matrix,camera_dist_matrix, image):

  dist = np.sqrt((tvec**2).sum()) # Calculates Distance

  # Calculate center point of input image
  im_width,im_height = image.shape[:2]
  start_point = (int(im_height/2), int(im_width/2))

  # Calculates 3D location of marker in camera and projects on 2D image plane
  end_point,_ = cv2.projectPoints(np.float32(np.array([[0,0,0]])), rvec, tvec, camera_matrix, camera_dist_matrix )
  end_point=np.int32(end_point[0][0]) # Stores result as integer

  # Calculate center point between start point and end point
  text_center = np.array(((start_point[0]+end_point[0])//2,(start_point[1]+end_point[1])//2))
  offset = 30
  text_center=text_center-30

  # Defines font, font scale, color, and thickness of text displayed
  font = cv2.FONT_HERSHEY_SIMPLEX  
  fontScale = 0.5
  color = (255, 0, 0)
  thickness = 2

  # Draws line on input image from center of image to marker
  image = cv2.line(image, start_point, end_point, (0,0,255) , thickness=2)
  
  # Puts text on image
  text = str(dist.round(2))
  image = cv2.putText(image, text, text_center, font, fontScale, color, thickness, cv2.LINE_AA)
  return image


def loadMarkers(markers):
  for marker in markers:
    marker_img = cv2.imread(marker['marker_file'])
    marker['sig'] = load_marker_dictionary(marker_img, bits=49)



class Fiducial_detector:
  # Class for foducial detector object that can be used to analyze image frames to give the camera offset and distance from the origin

  def __init__(self, log=True):
    
    # Flagger object
    self.fid_flagger = flagger.Flagger(
    config.gameplay['OPPONENT_TEAM_COLOR'], config.gameplay['HSV_LOWER_RED'], config.gameplay['HSV_UPPER_RED'], config.gameplay['HSV_LOWER_RED_2'], config.gameplay['HSV_UPPER_RED_2'], config.gameplay['HSV_LOWER_BLUE'], config.gameplay['HSV_UPPER_BLUE'])

    camera_intrinsic_matrix = config.fiducials['camera_intrinsic_matrix']
    camera_distortion_coefficients = config.fiducials['camera_distortion_coefficients']

    # Matrices used for camera calibration
    self.camera_matrix = np.array([
      camera_intrinsic_matrix['camera_focal_length_x'],
      0,
      camera_intrinsic_matrix['camera_center_x'],
      0,
      camera_intrinsic_matrix['camera_focal_length_y'],
      camera_intrinsic_matrix['camera_center_y'],
      0,
      0,
      1
        ]).reshape(3,3)
    self.camera_distortion_matrix = np.array([[
      camera_distortion_coefficients['camera_dist_coeff_k1'], 
      camera_distortion_coefficients['camera_dist_coeff_k2'], 
      camera_distortion_coefficients['camera_dist_coeff_p1'], 
      camera_distortion_coefficients['camera_dist_coeff_p2']] ])
    
    # Extract markers and loads them
    self.markers = config.fiducials['markers']
    self.markers = self.loadMarkers(self.markers)

    ms = config.fiducials['marker_size']

    self.marker_coords_rw = { #markers coordinates at 4 different rotations in real world
      0: np.float32(np.array([(0,0,0), (ms['width'],0,0), (ms['width'],ms['height'],0), (0,ms['height'],0)])) ,
      1: np.float32(np.array([(0,ms['height'],0),(0,0,0), (ms['width'],0,0), (ms['width'],ms['height'],0)])),
      2: np.float32(np.array([(ms['width'],ms['height'],0), (0,ms['height'],0), (0,0,0), (ms['width'],0,0) ])),
      3: np.float32(np.array([(ms['width'],0,0), (ms['width'],ms['height'],0), (0,ms['height'],0), (0,0,0)]))

    }

    #orgin and three points along three different axes(X,Y,Z) system of axes(and thus the orgin of this system) 
    #thus produced would be located at the centre of the fiducial. 
    self.axis_points = np.float32(np.array([
      (-ms['width']/2,-ms['height']/2,0),
      (ms['width']/2,-ms['height']/2,0),
      (0,ms['height']/2,0),
      (-ms['width']/2,-ms['height']/2,ms['height'])
    ]))

    #flag to log and visualize feed 
    self.log = log

    #These are transformation matrices. A transform matrix for a particular marker is used to get a point's coordinates in the real world coordinate
    #system from the coordinate system located at the center of the marker
    self.tx_matrices = self.getTxMatrices()

  #generates the transformation matrix based on the location of the markers in the real world coordinate system
  # math is base on the article written here: https://www.tangramvision.com/blog/rotate-scale-translate-coordinate-frames-for-multi-sensor-systems-part-2
  def transform(self, Tx, Ty, Tz, o, p, k):
      R = np.array([
          [cos(p)*cos(k), -cos(p)*sin(k), sin(p), Tx],
          [cos(o)*sin(k) + cos(k)*sin(o)*sin(p), cos(o)*cos(k)-sin(o)*sin(p)*sin(k), -cos(p)*sin(o), Ty],
          [sin(o)*sin(k)-cos(o)*cos(k)*sin(p), cos(k)*sin(o)+cos(o)*sin(p)*sin(k) , cos(o)*cos(p), Tz],
          [0,0,0,1]
      ])
      
      return R

  #obtains transformation matrices for all markers
  def getTxMatrices(self):
    txMatrices = {}

    for marker in fid_locs.locs:
      _matrix = self.transform(**fid_locs.locs[marker])
      txMatrices[marker] = _matrix
    return txMatrices
  


  def detectFiducial(self, frame, delta=1):
    print("--------------")
    hits, res_contours, res_image, detected_color= self.findMarker(frame, threshold=delta)
    

    if len(hits) == 0:
      res = {
        "success" : False
      }

      return res

    detected_marker_char = hits[-1][0]

    # Estimate pose of marker in 3D space 
    _, rvec, tvec = cv2.solvePnP(self.marker_coords_rw[hits[-1][2]] + np.array([-76.2,-76.2,0]) , res_contours, self.camera_matrix, self.camera_distortion_matrix)
    
    # Projects 3D axis points onto image plane 
    img_points,_ = cv2.projectPoints(self.axis_points, rvec, tvec, self.camera_matrix, self.camera_distortion_matrix)

    print("Camera Coordinate System (After solvePNP)")
    print("TVEC:",tvec,"RVEC:",rvec)

    # Converts img_points to integer values and reshapes into 4x2 array. 
    # Draws line between first point in img_points and the other three points 
    if self.log:
      img_points = np.int32(img_points.reshape((4,2)))

      res_image = cv2.line(res_image, img_points[0],img_points[1],(255,0,0),2)
      res_image = cv2.line(res_image, img_points[0],img_points[2],(0,255,0),2)
      res_image = cv2.line(res_image, img_points[0],img_points[3],(0,0,255),2)

    font = cv2.FONT_HERSHEY_SIMPLEX  
    fontScale = 1.5
    color = (255, 0, 0)
    thickness = 2

    # Computes rotation matrix
    rodM = cv2.Rodrigues(rvec)

    rotM = np.hstack([rodM[0], tvec])
    rotM = np.vstack([rotM, np.array([0,0,0,1])])

    invRotM = np.linalg.inv(rotM)

    text = "Translation Cam as origin: "+str(tvec[0].round(2)) + ',' +str(tvec[1].round(2))+' '+str(tvec[2].round(2))
    res_image = cv2.putText(res_image, text, (200,350) ,font, fontScale, color, thickness, cv2.LINE_AA)

    text = "Translation Marker as origin: "+str(invRotM[:,3][0].round(2)) + ',' +str(invRotM[:,3][1].round(2))+' '+str(invRotM[:,3][2].round(2))
    res_image = cv2.putText(res_image, text, (200,400) ,font, fontScale, color, thickness, cv2.LINE_AA)

    R = invRotM[:3,:3]

    translation_cm_wrt_marker_vector = invRotM[:,3]
    

    camera_plane = np.array([
      [-10,-10,0, 1],
      [-10,10,0, 1],
      [10,10,0, 1],
      [10,-10,0, 1]
    ]).T

    camera_plane_wrt_marker = invRotM@camera_plane

    # Extract the Euler angles from the rotation matrix
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
    yaw =np.degrees( np.arctan2(R[1, 0], R[0, 0]))

    print("Marker Coordinate System(After inversion)")
    print("tvec", translation_cm_wrt_marker_vector, "Roll {}, Pitch {}, Yaw {}".format(roll, pitch, yaw))
    print("camera_plane",camera_plane_wrt_marker)

    text = "Rotation: " + str(roll.round(2)) + ',' +str(pitch.round(2)) + ',' + str(yaw.round(2))
    res_image = cv2.putText(res_image, text, (200,550) ,font, fontScale, color, thickness, cv2.LINE_AA)


    text = "Color: " + detected_color
    res_image = cv2.putText(res_image, text, (200,650) ,font, fontScale, color, thickness, cv2.LINE_AA)

    # Draw Line
    res_image = draw_t_offset_line(rvec, tvec, self.camera_matrix,self.camera_distortion_matrix, res_image)
    
    try:
      tvec = np.append(tvec,np.array([[1]]))
      
      tx_matrix = self.tx_matrices[detected_marker_char]
      rw_loc_cam = tx_matrix@translation_cm_wrt_marker_vector
      camera_lane_rw = tx_matrix*camera_plane_wrt_marker

      print("Real World Coordinate System")
      print("Cam Loc", rw_loc_cam)
      print("Camera RW", camera_lane_rw)
    except Exception as e:
      print(e)
      print(tx_matrix)
      print(translation_cm_wrt_marker_vector)
      


    cv2.imshow('ROBOMASTER', res_image)
    cv2.waitKey(20)

    
    if len(hits)>0:
      detected_marker_char = hits[-1][0]
    else:
      detected_marker_char = ''

    #rw_orient_cam, n = orientWRTaxes(camera_plane_rw)

    res = {
      'success' : True,
      'rw_loc_cam' : rw_loc_cam,
      'mark_orient_cam': [roll, pitch, yaw],
      'detected_marker_char':detected_marker_char,
      'detected_color': detected_color
    }

    return res
    


  #given the signature of the detected fiducial, a match is found among the dictionary of established fiducials
  def matchMarker(self, image_signal, threshold):
    hits = []
    for marker in self.markers:
      for m_signal_tuple in marker['sig']:
        m_signal = np.array(m_signal_tuple[0])
        m_orientation_index = m_signal_tuple[1]
        n_misses = np.abs(np.sum((m_signal-image_signal)*m_signal))

        if self.log:
          #print("Total no. of correct bits:",m_signal.sum(),"Misses:",n_misses, "Threshold:",(int(threshold*m_signal.sum())))
          pass

        if (m_signal.sum()-n_misses)>=int(threshold*m_signal.sum()):
          hits.append((marker['marker_id'],m_signal.sum()-n_misses, m_orientation_index))
          
    if len(hits)==1:
      detectedChar = hits[0][0]
    elif len(hits)>1:
      #print("{} hits found, considering only the first".format(len(hits)))
      hits.sort(key = lambda x: x[1]) 
    #print('hits', hits)
    return hits



  def findMarker(self, img, threshold):

    bits = 49

    orig_img = img.copy()

    # Converts img to greyscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary Image with white regions for areas of interest
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    # Finds contours in binary image
    contours,_ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if self.log:
      #print("No. of contours:",len(contours))
      pass

    res_contours = []

    # Displays original image if no contours found 
    if len(contours)==0:
      if self.log:
        cv2.imshow('ROBOMASTER', orig_img)
        cv2.waitKey(20)

      return [],[],None,''
    

    for cnt in contours:

      # Simplify Contour
      poly_cnt = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True), True)
      
      # Checks if poly_cnt has 4 vertices, is convex, and has area greater than 1/1024
      if(len(poly_cnt)!=4 or not(cv2.isContourConvex(poly_cnt)) or cv2.contourArea(poly_cnt)<int((1/1024)*(img_thresh.shape[0]*img_thresh.shape[1])) ) :
        continue
      
      # Converts contoour points to float32 type
      poly_cnt = np.float32(poly_cnt)

      # Specifies termination criteria
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 40, 0.01)

      # Refine corner of contour 
      poly_cnt = cv2.cornerSubPix(img_thresh, poly_cnt, (5,5), (-1,-1),criteria)

      # Orders countour points 
      poly_cnt = order_contour(poly_cnt)

      # Adds to list
      res_contours.append(np.int32(poly_cnt.reshape((4,2))))

    # Check if contour successfully detected
    if len(res_contours)==0:
      if self.log:
        cv2.imshow('ROBOMASTER', orig_img)
        cv2.waitKey(20)
      return [],[],None,''
    
    # If multiple contour detected prints message that only first will be used
    if len(res_contours)>1:
      print('{} contours detected, considering only the first'.format(len(res_contours)))
    
    # Converts to ordered and refined contour points of first contour as float32
    res_contours = np.float32(res_contours[0])

    # Converts to NumPy array
    image_signal, warped_image = get_contour_bits(orig_img, res_contours, bits)
    image_signal = np.array(image_signal)

    # Determine color of marker
    detected_color = self.fid_flagger.red_or_blue(warped_image)

    # Compares marker to marker patterns
    hits = self.matchMarker(image_signal, threshold)

    # Checks if any mathces
    if len(hits)>0:
      detected_marker_char = hits[-1][0]
    else:
      detected_marker_char = ''

    # Set result image to the input image
    res_image = img

    if self.log:
      res_image = cv2.drawContours(orig_img, [np.int64(res_contours)],-1,(255,255,0),3)
      font = cv2.FONT_HERSHEY_SIMPLEX  
      fontScale = 10
      color = (255, 0, 0)
      thickness = 2
    
      res_image = cv2.putText(res_image,detected_marker_char,(200,250) ,font, fontScale, color, thickness, cv2.LINE_AA)
      
    return hits, res_contours, res_image, detected_color
    
  def loadMarkers(self, markers):
    for marker in markers:
      marker_img = cv2.imread(marker['marker_file'])
      marker['sig'] = load_marker_dictionary(marker_img, bits=49)

    return markers


#https://www.tangramvision.com/blog/rotate-scale-translate-coordinate-frames-for-multi-sensor-systems-part-2




