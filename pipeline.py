### Project 4: Advanced Lane Line finding

# Steps the final pipeline goes through:
# 1. Camera Calibration
# 2. Distortion Correction
# 3. Different Thresholds
# 4. Perspective Transformation
# 5. Draw Lines
# 6. Measure Curvature
# 7. Draw on image

# Start by importing all the packages needed

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import imageio

from moviepy.editor import VideoFileClip
from IPython.display import HTML


### 1. Camera Calibration

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
print("done")

### 2. Distortion Correction

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
matrix = {"mtx":mtx, "dist": dist}
pickle.dump(matrix,open("matrix.p","wb"))
print("Matrix created and saved!")

### 3. Functions for Different Thresholds

# Create classes for the lane lines

class Lane_Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # recent polynomial coefficients
        self.leftFit = [] 
        self.rightFit = []
        # line points to plot
        self.left_line_pts = []
        self.right_line_pts = []
        self.leftx = []
        self.lefty = []
        self.righty = []
        self.rightx = []
        self.count = 0
# Function to undistord images

def undistord(img):#
    matrix = pickle.load(open("matrix.p","rb"))
    mtx, dist = matrix["mtx"], matrix["dist"]
    output = cv2.undistort(img, mtx, dist, None, mtx)
    return output


# Sobel, magnitude and direction thresholding function

def multi_threshold(img, sobel_kernel=3, thresh_graddir=(0, np.pi/2), thresh_sobelx=(0, 255), thresh_mag=(0, 255)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Create binary thresholded image for the gradient direction
    binary_graddir =  np.zeros_like(absgraddir)
    binary_graddir[(absgraddir >= thresh_graddir[0]) & (absgraddir <= thresh_graddir[1])] = 1
    
    # Create binary thresholded image for sobel in x
    abs_sobelx = np.absolute(sobelx)
    
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    binary_sobelx = np.zeros_like(scaled_sobelx)
    binary_sobelx[(scaled_sobelx >= thresh_sobelx[0]) & (scaled_sobelx <= thresh_sobelx[1])] = 1
     
    # Create binary thresholded image for the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    
    binary_gradmag = np.zeros_like(gradmag)
    binary_gradmag[(gradmag >= thresh_mag[0]) & (gradmag <= thresh_mag[1])] = 1

  
    nadd = binary_sobelx + binary_gradmag + binary_graddir
    combined = np.zeros_like(gradmag)
    combined[nadd > 1]=1
    # Return the binary image
    return combined

# Color-thresholding functions

def gray_threshold(img, thresh_gray=(0,255)):
    # Grayscale threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_gray = np.zeros_like(gray)
    binary_gray[(gray > thresh_gray[0]) & (gray <= thresh_gray[1])] = 1
    return binary_gray

def red_threshold(img, thresh_red=(0,255)):
    # Grayscale threshold
    red = img[:,:,2]
    binary_r = np.zeros_like(red)
    binary_r[(red > thresh_red[0]) & (red <= thresh_red[1])] = 1
    return binary_r

def s_threshold(img, thresh_s=(0,255)):
 
    # Saturation threshold
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    binary_s = np.zeros_like(s_channel)
    binary_s[(s_channel > thresh_s[0]) & (s_channel <= thresh_s[1])] = 1
    return binary_s

# Combine the three thresholding functions

def final_thresh(img):
    n1 = multi_threshold(img,thresh_graddir=(0.6, 1.4),thresh_sobelx=(10, 110),thresh_mag=(20, 100))
    n2 = gray_threshold(img,thresh_gray=(220, 255))
    n4 = red_threshold(img, thresh_red=(200,255))
    n3 = s_threshold(img,thresh_s=(125, 255))
    nadd = n1+n2+n3+n4
    binary_out = np.zeros_like(n1)
    binary_out[nadd > 1]=1
    return binary_out

def region_of_interest(img):

    # creating a verice to cut out
    imshape = img.shape
    x_mid = imshape[1]/2
    y_mid = 480
    
    vertices = np.array([[(200,imshape[0]),(480, y_mid), (780, y_mid), (1200, imshape[0])]], dtype=np.int32)
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image

    ignore_mask_color = 1    # Binary output
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

### 4. Functions for Perspective Transformation

# Define perspective transformation function

def perspective(img):
    y_mid = 480
    gray = img
    img_size = (gray.shape[1], gray.shape[0])
    dst = np.float32([[80, 450], [img_size[0]-80, 450], [img_size[0]-80, 700], [80, 700]])
    src = np.float32([[479,y_mid],[799,y_mid],[1200,630],[80,630]])   # Source image is croped 
    M = cv2.getPerspectiveTransform(src, dst)
    wraped = cv2.warpPerspective(img, M, (img_size), flags=cv2.INTER_LINEAR)
    return wraped

def perspectiveInv(img):
    y_mid = 480
    gray = img
    img_size = (gray.shape[1], gray.shape[0])
    src = np.float32([[80, 450], [img_size[0]-80, 450], [img_size[0]-80, 700], [80, 700]])
    dst = np.float32([[479,y_mid],[799,y_mid],[1200,630],[80,630]])   # Source image is croped 
    M = cv2.getPerspectiveTransform(src, dst)
    wraped = cv2.warpPerspective(img, M, (img_size), flags=cv2.INTER_LINEAR)
    return wraped

### Final pipeline for images

def pipeline(img):
    binary_warped = img
    binary_warped = undistord(binary_warped)
    binary_warped = final_thresh(binary_warped)
    binary_warped = region_of_interest(binary_warped)
    binary_warped = perspective(binary_warped)
    if line.count == 0:
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # to avoid empty arrays use previous data
    if leftx.size == False:
        leftx = line.leftx
    if lefty.size == False:
        lefty = line.lefty
    if rightx.size == False:
        rightx = line.rightx
    if righty.size == False:
        righty = line.righty
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
        
    n = len(line.leftFit)
    if n>10:
        n = 10
                            
    line.leftFit.append(left_fit)
    line.rightFit.append(right_fit)
        
    best_left_fit = np.mean(line.leftFit[-n:], axis = 0)
    best_right_fit = np.mean(line.rightFit[-n:], axis = 0)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = best_left_fit[0]*ploty**2 + best_left_fit[1]*ploty + best_left_fit[2]
    right_fitx = best_right_fit[0]*ploty**2 + best_right_fit[1]*ploty + best_right_fit[2]
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Calculate position of the car to center of the line
    val = img.shape[0] * ym_per_pix
    leftLinePos = (left_fit_cr[0]*val**2)+(left_fit_cr[1]*val)+left_fit_cr[2]
    rightLinePos = (right_fit_cr[0]*val**2)+(right_fit_cr[1]*val)+right_fit_cr[2]
    lineMid = (leftLinePos+rightLinePos)/2
    CarPos = img.shape[1]*xm_per_pix/2
    disFromCenter = CarPos - lineMid
    centerText = "Car distance from center of the lane: {} m".format(abs(round(disFromCenter,2)))
    leftText = "Left line radius: {} m".format(round(left_curverad,2))
    rightText = "Right line radius: {} m".format(round(right_curverad,2))
    
    ### Visualize
    # Print angel and position on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, centerText, (10,50), font, 1,(255,100,100),2)
    cv2.putText(img, leftText, (10,100), font, 1,(255,100,100),2)
    cv2.putText(img, rightText, (10,150), font, 1,(255,100,100),2)

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin/4, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin/4, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    pts = np.hstack((left_line_window1, right_line_window2))
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)

    draw_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(draw_warp, np.int_([pts]), (0,255, 0))
    

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    final_warp = perspectiveInv(draw_warp)
    
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, final_warp, 0.3, 0)
    
    line.detected = True
    line.leftx = leftx
    line.lefty = lefty
    line.righty = righty
    line.rightx = rightx
    return result

    
### Last step: Apply pipeline to project video

line = Lane_Line()

video_output = 'laneLineText.mp4'
clip1 = VideoFileClip("project_video.mp4")
video_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
%time video_clip.write_videofile(video_output, audio=False)
