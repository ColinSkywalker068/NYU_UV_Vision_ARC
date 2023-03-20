import cv2
import math
import numpy as np

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
  
  return res

def loadMarkerDictionary(marker,bits):
  sigs = []
  w = len(marker[0])
  h= len(marker)
  # im_marker = cv2.cvtColor(marker,cv2.COLOR_BGR2GRAY)
  im_marker = marker.copy()
  cnt = np.array([[0,0],[w,0],[w,h],[0,h]],np.float32)
  for i in range(4):
    sig = getContourBits(im_marker,cnt,bits)
    sigs.append(sig)
    im_marker = cv2.rotate(im_marker, cv2.ROTATE_90_CLOCKWISE)

  return sigs


def findMarker(img, markers, threshold=0.5, bits=49, log=True):
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
    return []

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
    return []

  
  if log: 
    img_contours = cv2.drawContours(orig_img, [res_contours[0]],-1,(255,255,0),3)
    print("Final Contour:")
    cv2.imshow(img_contours)
  
  res_contours = np.float32(res_contours[0])

  image_sig = np.array(getContourBits(orig_img, res_contours, bits))
  hits = []
  for marker in markers:
    for sig in marker['sig']:
      sig = np.array(sig)
      n_misses = np.abs(np.sum((sig-image_sig)*sig))
      #print("Total no. of correct bits:",sig.sum(),"Misses:",n_misses, "Threshold:",(int(threshold*sig.sum())))

      if (sig.sum()-n_misses)>=int(threshold*sig.sum()):
        hits.append((marker['marker_id'],sig.sum()-n_misses))

  detectedChar = ''
  if len(hits)==1:
    detectedChar = hits[0][0]
  else:
    hits.sort(key = lambda x: x[1]) 
    detectedChar = hits[-1][0]
  
  print(res_contours)
  res_image = cv2.drawContours(orig_img, [np.int64(res_contours)],-1,(255,255,0),3)
  font = cv2.FONT_HERSHEY_SIMPLEX  
  fontScale = 10
  color = (255, 0, 0)
  thickness = 2
   
  res_image = cv2.putText(res_image,detectedChar,(200,250) ,font, fontScale, color, thickness, cv2.LINE_AA)
   
  cv2.imshow('ROBOMASTER', res_image)
  cv2.waitKey(20)

  return hits



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
    loadMarkers(markers) 
    vid_cap = cv2.VideoCapture(0)
    while(True):
        ret,frame = vid_cap.read()
        if ret:
            res = findMarker(frame, markers, threshold=1, bits=49, log=False)
            print(res)