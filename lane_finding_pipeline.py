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
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
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
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
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

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

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

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit, ploty

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

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
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, int([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, int([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return result

def measure_curvature_pixels(ploty, left_fit, right_fit):
    
    #Calculates the curvature of polynomial functions in pixels.  
    ym_per_pix = 20/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/880 # meters per pixel in x dimension

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
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

##the code below is the pipeline 

# Load the camera calibration specs calculated from CameraCal.py
CAMERA = pickle.load(open('calibrarion.p','rb'))
mtx = CAMERA['mtx']
dist = CAMERA['dist']
print(mtx)
imgPath = os.listdir("test_images/")
img = mpimg.imread("test_images/"+imgPath[1])
#Apply a distortion correction to raw images.
undist = cv2.undistort(img, mtx, dist, None, mtx)

#Use color transforms, gradients, etc., to create a thresholded binary image.
threshold_binary_img = threshold_binary(undist)
#default thresholds are s_thresh=(170, 255), sx_thresh=(20, 100)
warped_binary, m, Minv = warp(threshold_binary_img)


img_size = (img.shape[1], img.shape[0])  

src_pts = np.array([[580, 460], [710, 460], [1125, 720], [175, 720]]) #points of source
src_pts = np.int32([src_pts])
dest_pts = np.array([[200, 0], [1080, 0], [1080, 720], [200, 720]]) #points of destination
dest_pts = np.int32([dest_pts])
warped_undist, m_undist, Minv_undist = warp(undist)
src_pts_img = undist.copy()
dest_pts_img = warped_undist.copy()
#pts = np.int32([pts])
cv2.polylines(src_pts_img, src_pts, True, (0,5,150), thickness = 5)
cv2.polylines(dest_pts_img, dest_pts, True, (0,5,150), thickness = 5)

# Plot source image & # Plot destination image
fig_src_dst_pts = plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.imshow(src_pts_img)
plt.title("Source Points")
plt.subplot(1, 2, 2)
plt.imshow(dest_pts_img)
plt.title("Destination Points")
fig_src_dst_pts.savefig('output_images/birds-eye view.png')

# Plot binary image
fig_binary= plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.imshow(undist)
plt.title("undistorted image")
plt.subplot(1, 2, 2)
plt.imshow(threshold_binary_img)
plt.title("binary image")
fig_binary.savefig('output_images/binary.png')

#Detect lane pixels and fit to find the lane boundary.
leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped_binary)
out_img, left_fit, right_fit, ploty = fit_polynomial(warped_binary)
#calculate the curvature 
left_curverad, right_curverad = measure_curvature_pixels(ploty, left_fit, right_fit)
print('left curvature is',left_curverad,'/n','right curvature is ', right_curverad)
position = position_from_center(ploty, left_fit, right_fit)
print('position is ', position)
plt.figure(figsize=(20,10))
plt.imshow(out_img)
plt.show()
#plt.close()

