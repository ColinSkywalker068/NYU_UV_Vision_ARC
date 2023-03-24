import cv2
import math
import numpy as np
import flagger 
import config

def orderContour(cnt):
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


def getContourBits(im,cnt,bits):

  
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

def loadMarkerDictionary(marker,bits):
  sigs = []
  w = len(marker[0])
  h= len(marker)
  # im_marker = cv2.cvtColor(marker,cv2.COLOR_BGR2GRAY)
  im_marker = marker.copy()
  cnt = np.array([[0,0],[w,0],[w,h],[0,h]],np.float32)
  for i in range(4):
    sig,_ = getContourBits(im_marker,cnt,bits)
    sigs.append((sig,i))
    im_marker = cv2.rotate(im_marker, cv2.ROTATE_90_CLOCKWISE)

  return sigs


def findMarker(img, markers, threshold=0.5, bits=49, log=True, flagger_obj=None):
  orig_img = img.copy()
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)

  if log:
    print("After Thresholding")
    cv2.imshow(img_thresh)
  contours,_ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  if log:
    print("No. of contours:",len(contours))
    img_contours = cv2.drawContours(img, contours,-1,(255,255,0),3)
    cv2.imshow(img_contours)
  res_contours = []

  if len(contours)==0:
    cv2.imshow('ROBOMASTER', orig_img)
    cv2.waitKey(20)
    return [],[],-1,None,''
  

  for cnt in contours:
    poly_cnt = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True), True)
    
    if(len(poly_cnt)!=4 or cv2.contourArea(poly_cnt)<int((1/16)*(img_thresh.shape[0]*img_thresh.shape[1]))):
      continue

    poly_cnt = np.float32(poly_cnt)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 40, 0.01)
    poly_cnt = cv2.cornerSubPix(img_thresh, poly_cnt, (5,5), (-1,-1),criteria)
    poly_cnt = orderContour(poly_cnt)
    res_contours.append(np.int32(poly_cnt.reshape((4,2))))

  if len(res_contours)==0:
    cv2.imshow('ROBOMASTER', orig_img)
    cv2.waitKey(20)
    return [],[],-1,None,''

  
  if log: 
    img_contours = cv2.drawContours(orig_img, [res_contours[0]],-1,(255,255,0),3)
    print("Final Contour:")
    cv2.imshow(img_contours)
  
  res_contours = np.float32(res_contours[0])

  image_sig, warped_image = getContourBits(orig_img, res_contours, bits)
  image_sig = np.array(image_sig)
  hits = []

  detected_color = flagger_obj.red_or_blue(warped_image)


  hits_index = []
  for marker in markers:
    for sig_ in marker['sig']:
      sig = np.array(sig_[0])
      n_misses = np.abs(np.sum((sig-image_sig)*sig))
      #print("Total no. of correct bits:",sig.sum(),"Misses:",n_misses, "Threshold:",(int(threshold*sig.sum())))

      if (sig.sum()-n_misses)>=int(threshold*sig.sum()):
        hits.append((marker['marker_id'],sig.sum()-n_misses))
        hits_index.append(sig_[1])
        

  detectedChar = ''
  if len(hits)==1:
    detectedChar = hits[0][0]
  elif len(hits)>1:
    hits.sort(key = lambda x: x[1]) 
    detectedChar = hits[-1][0]
  
  res_image = cv2.drawContours(orig_img, [np.int64(res_contours)],-1,(255,255,0),3)
  font = cv2.FONT_HERSHEY_SIMPLEX  
  fontScale = 10
  color = (255, 0, 0)
  thickness = 2
   
  res_image = cv2.putText(res_image,detectedChar,(200,250) ,font, fontScale, color, thickness, cv2.LINE_AA)
   
  

  return hits,res_contours,hits_index,res_image,detected_color


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
    marker['sig'] = loadMarkerDictionary(marker_img, bits=49)



if __name__=='__main__':
    markers = [
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

    
    ]

    flagger_obj = flagger.Flagger(
    config.gameplay['OPPONENT_TEAM_COLOR'], config.gameplay['HSV_LOWER_RED'], config.gameplay['HSV_UPPER_RED'], config.gameplay['HSV_LOWER_RED_2'], config.gameplay['HSV_UPPER_RED_2'], config.gameplay['HSV_LOWER_BLUE'], config.gameplay['HSV_UPPER_BLUE'])


    camera_focal_length_x = 1427.9641708247764
    camera_center_x =  974.409732316623
    camera_focal_length_y = 1434.9326534580543
    camera_center_y =  571.1514876178345

    # Distortion coefficients.
    # See the OpenCV documentation on what this does before adjusting.
    camera_dist_coeff_k1 = -0.07231168346387425
    camera_dist_coeff_k2 = 0.054653045623148634
    camera_dist_coeff_p1 = 0.013779335161555138
    camera_dist_coeff_p2 = 0.005759852619212033
    camera_dist_coeff_k3 = 0.15626401335130446

    camera_matrix = np.array([camera_focal_length_x,0, camera_center_x,0, camera_focal_length_y, camera_center_y,0,0,1]).reshape(3,3)
    camera_dist_matrix = np.array([[camera_dist_coeff_k1, camera_dist_coeff_k2, camera_dist_coeff_p1, camera_dist_coeff_p2] ])

    loadMarkers(markers) 
    vid_cap = cv2.VideoCapture(0)

    marker_coords = {
      0: np.float32(np.array([(0,0,0), (152.4,0,0), (152.4,152.4,0), (0,152.4,0)])) ,
      1: np.float32(np.array([(0,152.4,0),(0,0,0), (152.4,0,0), (152.4,152.4,0)])),
      2: np.float32(np.array([(152.4,152.4,0), (0,152.4,0), (0,0,0), (152.4,0,0) ])),
      3: np.float32(np.array([(152.4,0,0), (152.4,152.4,0), (0,152.4,0), (0,0,0)]))

    }
    axis = np.float32(np.array([
      (0,0,0),
      (152.4,0,0),
      (0,152.4,0),
      (0,0,152.4)
    ])) + np.array([-76.2,-76.2,0])
    while(True):
        ret,frame = vid_cap.read()
        if ret:
            hits, res_contours, hits_index, res_image, detected_color= findMarker(frame, markers, threshold=1, bits=49, log=False, flagger_obj = flagger_obj)

            if len(hits)==0:
              continue

            _, rvec, tvec = cv2.solvePnP(marker_coords[hits_index[-1]] + np.array([-76.2,-76.2,0]) , res_contours, camera_matrix, camera_dist_matrix)
            img_points,_ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_dist_matrix )

            print(rvec)

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

            res_image = draw_t_offset_line(rvec, tvec, camera_matrix,camera_dist_matrix, res_image)

            # rotM = cv2.Rodrigues(rvec)[0]
            # cameras_position = -np.matrix(rotM).T * np.matrix(tvec)
            # res_image = cv2.putText(res_image, str(cameras_position), (200,550) ,font, fontScale, color, thickness, cv2.LINE_AA)


            cv2.imshow('ROBOMASTER', res_image)
            cv2.waitKey(20)





