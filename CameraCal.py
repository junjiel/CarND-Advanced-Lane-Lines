import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

###find the image points and object points, calculate the camera calibration matrix and distortion coefficients
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) #x, y coordinates 

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    cal_img = cv2.imread(fname)
    cal_gray = cv2.cvtColor(cal_img,cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(cal_gray, (9,6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
    
        # Draw and display the corners
        cal_img = cv2.drawChessboardCorners(cal_img, (9,6), corners, ret)
       
        #cv2.waitKey(500)

cv2.destroyAllWindows()

###

###correct the distortion

img = cv2.imread('camera_cal/calibration1.jpg')
test_img = cv2.imread('test_images/test1.jpg')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
undistorted = cv2.undistort(img, mtx, dist, None, mtx)
test_undistorted = cv2.undistort(test_img, mtx, dist, None, mtx)

#plot and save undistorted  calibration image
f = plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("original")
plt.subplot(1, 2, 2)
plt.imshow(undistorted)
plt.title("undistorted")
f.savefig('output_images/CameraCalResult.png')

#plot and save undistorted  test image
f_test = plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.imshow(test_img)
plt.title("original")

plt.subplot(1, 2, 2)
plt.imshow(test_undistorted)
plt.title("undistorted")
f_test.savefig('output_images/undistorted_test_img.png')
plt.show()

# Save the fitted camera parameters
print(mtx)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "calibrarion.p", "wb" ) )




###The code block below plots results of camera calibration, uncomment it for plotting original and undistorted chessboard
#img = cv2.imread('camera_cal/calibration1.jpg')
#undistorted = cal_undistort(img, objpoints, imgpoints)
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()
#ax1.imshow(img)
#ax1.set_title('Original Image', fontsize=50)
#ax2.imshow(undistorted)
#ax2.set_title('Undistorted Image', fontsize=50)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#plt.show()
#f.savefig('output_images/CameraCalResult.png')
#plt.close()

