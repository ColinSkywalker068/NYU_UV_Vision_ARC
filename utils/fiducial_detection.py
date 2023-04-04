import cv2
import math
import numpy as np
import flagger 
import config


def order_contour(cnt):
  cx = (cnt[0][0][0] + cnt[1][0][0] + cnt[2][0][0] + cnt[3][0][0])/4.0
  cy = (cnt[0][0][1] + cnt[1][0][1] + cnt[2][0][1] + cnt[3][0][1])/4.0

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


def get_contour_bits(im,cnt,bits):

  
  pixelLen = int(math.sqrt(bits))

  corners = np.array([[0,0],[bits,0],[bits,bits],[0,bits]],np.float32)

  M = cv2.getPerspectiveTransform(cnt, corners)
  dst = cv2.warpPerspective(im, M, (bits, bits),flags=cv2.INTER_LINEAR)
  #cv2.imshow(dst) 

  gs =  cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
  ret, im_thres = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  #cv2.imshow(im_thres)

  element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
  im_er = cv2.erode(im_thres,element)
  #cv2.imshow(im_er)

  im_size=im_er.shape
  #print("Image Shape:",im_size)
  grid_cell_size = (im_size[0]//pixelLen, im_size[1]//pixelLen)
  #print("Grid Cell size:",grid_cell_size)

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
  sigs = []
  w = len(marker[0])
  h= len(marker)
  # im_marker = cv2.cvtColor(marker,cv2.COLOR_BGR2GRAY)
  im_marker = marker.copy()
  cnt = np.array([[0,0],[w,0],[w,h],[0,h]],np.float32)
  for i in range(4):
    sig,_ = get_contour_bits(im_marker,cnt,bits)
    sigs.append((sig,i))
    im_marker = cv2.rotate(im_marker, cv2.ROTATE_90_CLOCKWISE)

  return sigs


# def findMarker(img, markers, flagger_obj=None, threshold=0.5, bits=49, log=True):
#   orig_img = img.copy()
#   img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)

#   if log:
#     print("After Thresholding")
#     cv2.imshow(img_thresh)
#   contours,_ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#   if log:
#     print("No. of contours:",len(contours))
#     img_contours = cv2.drawContours(img, contours,-1,(255,255,0),3)
#     cv2.imshow(img_contours)
#   res_contours = []

#   if len(contours)==0:
#     cv2.imshow('ROBOMASTER', orig_img)
#     cv2.waitKey(20)
#     return [],[],-1,None,''
  

#   for cnt in contours:
#     poly_cnt = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True), True)
    
#     if(len(poly_cnt)!=4 or cv2.contourArea(poly_cnt)<int((1/16)*(img_thresh.shape[0]*img_thresh.shape[1]))):
#       continue

#     poly_cnt = np.float32(poly_cnt)
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 40, 0.01)
#     poly_cnt = cv2.cornerSubPix(img_thresh, poly_cnt, (5,5), (-1,-1),criteria)
#     poly_cnt = orderContour(poly_cnt)
#     res_contours.append(np.int32(poly_cnt.reshape((4,2))))

#   if len(res_contours)==0:
#     cv2.imshow('ROBOMASTER', orig_img)
#     cv2.waitKey(20)
#     return [],[],-1,None,''

  
#   if log: 
#     img_contours = cv2.drawContours(orig_img, [res_contours[0]],-1,(255,255,0),3)
#     print("Final Contour:")
#     cv2.imshow(img_contours)
  
#   res_contours = np.float32(res_contours[0])

#   image_sig, warped_image = getContourBits(orig_img, res_contours, bits)
#   image_sig = np.array(image_sig)
#   hits = []

#   detected_color = flagger_obj.red_or_blue(warped_image)


#   hits_index = []
#   for marker in markers:
#     for sig_ in marker['sig']:
#       sig = np.array(sig_[0])
#       n_misses = np.abs(np.sum((sig-image_sig)*sig))
#       #print("Total no. of correct bits:",sig.sum(),"Misses:",n_misses, "Threshold:",(int(threshold*sig.sum())))

#       if (sig.sum()-n_misses)>=int(threshold*sig.sum()):
#         hits.append((marker['marker_id'],sig.sum()-n_misses))
#         hits_index.append(sig_[1])
        

#   detectedChar = ''
#   if len(hits)==1:
#     detectedChar = hits[0][0]
#   elif len(hits)>1:
#     hits.sort(key = lambda x: x[1]) 
#     detectedChar = hits[-1][0]
  
#   res_image = cv2.drawContours(orig_img, [np.int64(res_contours)],-1,(255,255,0),3)
#   font = cv2.FONT_HERSHEY_SIMPLEX  
#   fontScale = 10
#   color = (255, 0, 0)
#   thickness = 2
   
#   res_image = cv2.putText(res_image,detectedChar,(200,250) ,font, fontScale, color, thickness, cv2.LINE_AA)
   
  

#   return hits,res_contours,hits_index,res_image,detected_color


def draw_t_offset_line(rvec, tvec, camera_matrix,camera_dist_matrix, image):
  dist = np.sqrt((tvec**2).sum())
  print(dist)
  im_width,im_height = image.shape[:2]
  
  start_point = (int(im_height/2), int(im_width/2))

  end_point,_ = cv2.projectPoints(np.float32(np.array([[0,0,0]])), rvec, tvec, camera_matrix, camera_dist_matrix )
  end_point=np.int32(end_point[0][0])

  text_center = np.array(((start_point[0]+end_point[0])//2,(start_point[1]+end_point[1])//2))
  offset = 30
  text_center=text_center-30

  font = cv2.FONT_HERSHEY_SIMPLEX  
  fontScale = 0.5
  color = (255, 0, 0)
  thickness = 2

  image = cv2.line(image, start_point, end_point, (0,0,255) , thickness=2)

  text = str(dist.round(2))
  image = cv2.putText(image, text, text_center, font, fontScale, color, thickness, cv2.LINE_AA)
  return image


def loadMarkers(markers):
  for marker in markers:
    marker_img = cv2.imread(marker['marker_file'])
    marker['sig'] = load_marker_dictionary(marker_img, bits=49)



class Fiducial_detector:
  def __init__(self, log=True):
    self.fid_flagger = flagger.Flagger(
    config.gameplay['OPPONENT_TEAM_COLOR'], config.gameplay['HSV_LOWER_RED'], config.gameplay['HSV_UPPER_RED'], config.gameplay['HSV_LOWER_RED_2'], config.gameplay['HSV_UPPER_RED_2'], config.gameplay['HSV_LOWER_BLUE'], config.gameplay['HSV_UPPER_BLUE'])

    camera_intrinsic_matrix = config.fiducials['camera_intrinsic_matrix']
    camera_distortion_coefficients = config.fiducials['camera_distortion_coefficients']

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
    
    self.markers = config.fiducials['markers']
    self.markers = self.loadMarkers(self.markers)

    ms = config.fiducials['marker_size']

    self.marker_coords_rw = { #markers coordinates at 4 different rotations in real world
      0: np.float32(np.array([(0,0,0), (ms['width'],0,0), (ms['width'],ms['height'],0), (0,ms['height'],0)])) ,
      1: np.float32(np.array([(0,ms['height'],0),(0,0,0), (ms['width'],0,0), (ms['width'],ms['height'],0)])),
      2: np.float32(np.array([(ms['width'],ms['height'],0), (0,ms['height'],0), (0,0,0), (ms['width'],0,0) ])),
      3: np.float32(np.array([(ms['width'],0,0), (ms['width'],ms['height'],0), (0,ms['height'],0), (0,0,0)]))

    }

    self.axis_points = np.float32(np.array([
      (-ms['width']/2,-ms['height']/2,0),
      (ms['width']/2,-ms['height']/2,0),
      (0,ms['height']/2,0),
      (-ms['width']/2,-ms['height']/2,ms['height'])
    ]))

    self.log = log

  def detectFiducial(self, frame, delta=1):
    hits, res_contours, res_image, detected_color= self.findMarker(frame, threshold=delta)

    if len(hits) == 0:
      res = {
        "success" : False
      }

      return res

    _, rvec, tvec = cv2.solvePnP(self.marker_coords_rw[hits[-1][2]] + np.array([-76.2,-76.2,0]) , res_contours, self.camera_matrix, self.camera_distortion_matrix)
    img_points,_ = cv2.projectPoints(self.axis_points, rvec, tvec, self.camera_matrix, self.camera_distortion_matrix)


    if self.log:
      img_points = np.int32(img_points.reshape((4,2)))

      res_image = cv2.line(res_image, img_points[0],img_points[1],(255,0,0),2)
      res_image = cv2.line(res_image, img_points[0],img_points[2],(0,255,0),2)
      res_image = cv2.line(res_image, img_points[0],img_points[3],(0,0,255),2)

      font = cv2.FONT_HERSHEY_SIMPLEX  
      fontScale = 1.5
      color = (255, 0, 0)
      thickness = 2

      text = "Translation: "+str(tvec[0].round(2)) + ',' +str(tvec[1].round(2))+' '+str(tvec[2].round(2))
      res_image = cv2.putText(res_image, text, (200,350) ,font, fontScale, color, thickness, cv2.LINE_AA)

      text = "Rotation: "+str(rvec[0].round(2)) + ',' +str(rvec[1].round(2))+' '+str(rvec[2].round(2))
      res_image = cv2.putText(res_image, text, (200,450) ,font, fontScale, color, thickness, cv2.LINE_AA)

      text = "Color: " + detected_color
      res_image = cv2.putText(res_image, text, (200,550) ,font, fontScale, color, thickness, cv2.LINE_AA)

      res_image = draw_t_offset_line(rvec, tvec, self.camera_matrix,self.camera_distortion_matrix, res_image)

      cv2.imshow('ROBOMASTER', res_image)
      cv2.waitKey(20)

        # rotM = cv2.Rodrigues(rvec)[0]
        # cameras_position = -np.matrix(rotM).T * np.matrix(tvec)
        # res_image = cv2.putText(res_image, str(cameras_position), (200,550) ,font, fontScale, color, thickness, cv2.LINE_AA)

    if len(hits)>0:
      detected_marker_char = hits[-1][0]
    else:
      detected_marker_char = ''

    res = {
      'success' : True,
      'translation' : tvec,
      'rotation' : rvec,
      'detected_marker_char':detected_marker_char,
      'detected_color': detected_color
    }

    return res
    


  def matchMarker(self, image_signal, threshold):
    hits = []
    for marker in self.markers:
      for m_signal_tuple in marker['sig']:
        m_signal = np.array(m_signal_tuple[0])
        m_orientation_index = m_signal_tuple[1]
        n_misses = np.abs(np.sum((m_signal-image_signal)*m_signal))

        if self.log:
          print("Total no. of correct bits:",m_signal.sum(),"Misses:",n_misses, "Threshold:",(int(threshold*m_signal.sum())))

        if (m_signal.sum()-n_misses)>=int(threshold*m_signal.sum()):
          hits.append((marker['marker_id'],m_signal.sum()-n_misses, m_orientation_index))
          
    if len(hits)==1:
      detectedChar = hits[0][0]
    elif len(hits)>1:
      print("{} hits found, considering only the first".format(len(hits)))
      hits.sort(key = lambda x: x[1]) 
    print('hits', hits)
    return hits



  def findMarker(self, img, threshold):

    bits = 49

    orig_img = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    contours,_ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if self.log:
      print("No. of contours:",len(contours))

    res_contours = []

    if len(contours)==0:
      if self.log:
        cv2.imshow('ROBOMASTER', orig_img)
        cv2.waitKey(20)

      return [],[],None,''
    

    for cnt in contours:

      poly_cnt = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True), True)
      
      if(len(poly_cnt)!=4 or cv2.contourArea(poly_cnt)<int((1/16)*(img_thresh.shape[0]*img_thresh.shape[1]))):
        continue

      poly_cnt = np.float32(poly_cnt)

      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 40, 0.01)
      poly_cnt = cv2.cornerSubPix(img_thresh, poly_cnt, (5,5), (-1,-1),criteria)

      poly_cnt = order_contour(poly_cnt)

      res_contours.append(np.int32(poly_cnt.reshape((4,2))))

    if len(res_contours)==0:
      if self.log:
        cv2.imshow('ROBOMASTER', orig_img)
        cv2.waitKey(20)
      return [],[],None,''
    
    if len(res_contours)>1:
      print('{} contours detected, considering only the first'.format(len(res_contours)))
    
    res_contours = np.float32(res_contours[0])

    image_signal, warped_image = get_contour_bits(orig_img, res_contours, bits)
    image_signal = np.array(image_signal)

    detected_color = self.fid_flagger.red_or_blue(warped_image)
    hits = self.matchMarker(image_signal, threshold)

    if len(hits)>0:
      detected_marker_char = hits[-1][0]
    else:
      detected_marker_char = ''

    if self.log:
      res_image = cv2.drawContours(orig_img, [np.int64(res_contours)],-1,(255,255,0),3)
      font = cv2.FONT_HERSHEY_SIMPLEX  
      fontScale = 10
      color = (255, 0, 0)
      thickness = 2
      
      print(detected_marker_char)
      res_image = cv2.putText(res_image,detected_marker_char,(200,250) ,font, fontScale, color, thickness, cv2.LINE_AA)
      
      

    return hits, res_contours, res_image, detected_color
    
  def loadMarkers(self, markers):
    for marker in markers:
      marker_img = cv2.imread(marker['marker_file'])
      marker['sig'] = load_marker_dictionary(marker_img, bits=49)

    return markers







