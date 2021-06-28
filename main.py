import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os



###all functions will be defined below

def warp(img):
    # Pass in your image into this function
    # Write code to do the following steps
    #img = np.copy(image)
    img_size = (img.shape[1], img.shape[0])  
    #offsetx = 325
    #offsety = 450
    # define 4 source points src = np.float32([[,],[,],[,],[,]])
    #src = np.float32([[280+offsetx,offsety],[1000 -offsetx, offsety],[1000, img_size[1]-45],[280,img_size[1]-45]])
    #src = np.array([[280+offsetx,offsety],[1000 -offsetx, offsety],[1080, img_size[1]],[200,img_size[1]]],np.float32) 
    src = np.array([[580, 460], [710, 460], [1125, 720], [175, 720]], np.float32)
    # define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    #dst = np.float32([[280, 0], [1000, 0], [1000, img_size[1]], [280, img_size[1]]])
    #dst = np.array([[330, 0], [950, 0], [950, img_size[1]], [330, img_size[1]]], np.float32) 
    dst = np.array([[200, 0], [1080, 0], [1080, 720], [200, 720]], np.float32) 
    #  use cv2.getPerspectiveTransform() to get M(perspective transform), Minv(inverse perspective transform) the transform matrix
    m= cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, m, img_size, flags = cv2.INTER_LINEAR)
    return warped, m, Minv

def threshold_binary(image, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(image)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #h_channel = hls[:,:,0]
    #l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Stack each channel, uncomment the code below if you want to display result from gradiant and color 
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        #nonzeroy stores [y1, y2,...yn], while nonzerox stores[x1, x2, ...xn]. 
        #The indices of nonzerox and nonzeroy are identical since they represent the same point.
        #For example, the index of y1 and x1 are the same, which represent point1. In the code below,
        #the logic operation will return a list of True False,the nonzero()function will
        #return the indices of Trues(they are represented as 1) in the nonzerox and nonzeroy. 
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0] 
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    return left_fit, right_fit, left_fitx, right_fitx, ploty

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fit, right_fit, left_fitx, right_fitx, ploty

def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = (nonzerox > (left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2] -margin)) & (nonzerox < (left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2] + margin))
                   
    right_lane_inds = (nonzerox >(right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2] -margin)) & (nonzerox < (right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2]+ margin))
                    
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    return left_fit, right_fit, left_fitx, right_fitx, ploty

def measure_curvature_pixels(ploty, left_fitx, right_fitx):
    
    #Calculates the curvature of polynomial functions in pixels.  
    ym_per_pix = 20/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/880 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_A = left_fit_cr[0]
    left_B = left_fit_cr[1]
    left_curverad = ((1 + (2*left_A*y_eval*ym_per_pix + left_B)**2)**1.5) / np.abs(2*left_A)  ## Implement the calculation of the left line here
    right_A = right_fit_cr[0]
    right_B = right_fit_cr[1]
    right_curverad = ((1 + (2*right_A*y_eval*ym_per_pix + right_B)**2)**1.5) / np.abs(2*right_A)   ## Implement the calculation of the right line here
    
    return left_curverad, right_curverad

def position_from_center(ploty, left_fit, right_fit):
    image_width = 1280
    xm_per_pix = 3.7/880
    #find the y value at the bottom of the image
    y_eval = np.max(ploty)
    #find the right and left x values at the bottom of the image
    leftx = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    rightx = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    position = np.abs((rightx-leftx)/2 - image_width/2)*xm_per_pix
    return position

def draw_line(warped, undist, left_fitx, right_fitx, ploty, Minv, left_radius, right_radius, position_from_center):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    #write text on image
    curvature_text = "Left Curvature is " + str(np.round(left_radius, 2)) + ", Right Curvature is " + str(np.round(right_radius, 2))
    font = cv2.FONT_HERSHEY_TRIPLEX    
    cv2.putText(result, curvature_text, (30, 60), font, 1, (0,255,0), 2)
    deviation_text = "Vehicle position from lane center is {:.2f} m".format(position_from_center) 
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(result, deviation_text, (30, 90), font, 1, (0,255,0), 2)

    return result

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None



imgPath = os.listdir("test_images/")
img = mpimg.imread("test_images/"+imgPath[1])
img = np.copy(img)
#create left and right line insctance of class Line() to store line info
left_line = Line()
right_line = Line()
#load the camera calibration data
CAMERA = pickle.load(open('calibrarion.p','rb'))
mtx = CAMERA['mtx']
dist = CAMERA['dist']
#Apply a distortion correction to raw images.
undist = cv2.undistort(img, mtx, dist, None, mtx)
#Use color transforms, gradients, etc., to create a thresholded binary image.
threshold_binary_img = threshold_binary(undist)
#default thresholds are s_thresh=(170, 255), sx_thresh=(20, 100), birds-eye view
warped_binary, m, Minv = warp(threshold_binary_img)

if (left_line.detected == False & right_line.detected == False):
    left_line.current_fit, right_line.current_fit, left_line.allx, right_line.allx, ploty = fit_polynomial(warped_binary)
else:
    left_line.current_fit, right_line.current_fit, left_line.allx, right_line.allx, ploty = search_around_poly(warped_binary, left_line.current_fit,right_line.current_fit)

left_line.radius_of_curvature,right_line.radius_of_curvature= measure_curvature_pixels(ploty,left_line.allx,right_line.allx)
left_line.line_base_pos = position_from_center(ploty,left_line.current_fit,right_line.current_fit)
visualization = draw_line(warped_binary,undist, left_line.allx,right_line.allx,ploty, Minv, left_line.radius_of_curvature,right_line.radius_of_curvature, left_line.line_base_pos)
f = plt.figure()
plt.imshow(visualization)
f.savefig('output_images/visualization.png')
plt.show()