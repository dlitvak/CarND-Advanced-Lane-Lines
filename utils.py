import cv2
import numpy as np


def abs_sobel(img, orient=(1, 0), sobel_kernel=3, thresh=(0, 255)):
    """Apply Sobel gradient in x/y orientation"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobel = cv2.Sobel(gray, cv2.CV_64F, orient[0], orient[1], ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    s_binary = np.zeros_like(scaled_sobel)
    s_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return s_binary


def color_select(img, thresh=(0, 255), color_conv_type=cv2.COLOR_RGB2HLS, color_conv_indx=2):
    color_cvt = img
    if color_conv_type is not None:
        color_cvt = cv2.cvtColor(img, color_conv_type)
    channel = color_cvt[:, :, color_conv_indx]

    binary_output = np.zeros_like(channel)
    binary_output[((channel > thresh[0]) & (channel <= thresh[1]))] = 1
    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobelxy = np.uint8(255 * sobelxy / np.max(sobelxy))

    mag_binary = np.zeros_like(scaled_sobelxy)
    mag_binary[(scaled_sobelxy >= thresh[0]) & (scaled_sobelxy <= thresh[1])] = 1

    return mag_binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)

    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobely = np.absolute(sobely)

    dir = np.arctan2(abs_sobely, abs_sobelx)

    dir_binary = np.zeros_like(sobelx)
    dir_binary[(dir >= thresh[0]) & (dir <= thresh[1])] = 1

    return dir_binary


# def skip_sliding_win(binary_warped_img, left_coeff=(), right_coeff=(), margin=100, min_inds_per_lane=3600,
#                      base_ctr=(0.0, 0.0)):
#     # Assume you now have a new warped binary image
#     # from the next frame of video (also called "binary_warped")
#     # It's now much easier to find line pixels!
#     nonzero = binary_warped_img.nonzero()
#     nonzeroy = np.array(nonzero[0])
#     nonzerox = np.array(nonzero[1])
#     left_lane_inds = ((nonzerox > (left_coeff[0] * (nonzeroy ** 2) + left_coeff[1] * nonzeroy +
#                                    left_coeff[2] - (base_ctr[0] - margin))) & (nonzerox < (left_coeff[0] * (nonzeroy ** 2) +
#                                                                            left_coeff[1] * nonzeroy + left_coeff[
#                                                                              2] + (base_ctr[0] + margin))))
#
#     right_lane_inds = ((nonzerox > (right_coeff[0] * (nonzeroy ** 2) + right_coeff[1] * nonzeroy +
#                                     right_coeff[2] - (base_ctr[1] + margin))) & (nonzerox < (right_coeff[0] * (nonzeroy ** 2) +
#                                                                              right_coeff[1] * nonzeroy + right_coeff[
#                                                                                2] + (base_ctr[1] + margin))))
#
#     # Again, extract left and right line pixel positions
#     leftx = nonzerox[left_lane_inds]
#     lefty = nonzeroy[left_lane_inds]
#     rightx = nonzerox[right_lane_inds]
#     righty = nonzeroy[right_lane_inds]
#     # Fit a second order polynomial to each
#     left_coeff = np.polyfit(lefty, leftx, 2)
#     right_coeff = np.polyfit(righty, rightx, 2)
#     # Generate x and y values for plotting
#     ploty = np.linspace(0, binary_warped_img.shape[0] - 1, binary_warped_img.shape[0])
#     left_fitx = left_coeff[0] * ploty ** 2 + left_coeff[1] * ploty + left_coeff[2]
#     right_fitx = right_coeff[0] * ploty ** 2 + right_coeff[1] * ploty + right_coeff[2]
#
#     # Create an image to draw on and an image to show the selection window
#     out_img = np.dstack((binary_warped_img, binary_warped_img, binary_warped_img)) * 255
#     window_img = np.zeros_like(out_img)
#     # Color in left and right line pixels
#     out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
#     out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
#
#     # Generate a polygon to illustrate the search window area
#     # And recast the x and y points into usable format for cv2.fillPoly()
#     left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
#     left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
#                                                                     ploty])))])
#     left_line_pts = np.hstack((left_line_window1, left_line_window2))
#     right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
#     right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
#                                                                      ploty])))])
#     right_line_pts = np.hstack((right_line_window1, right_line_window2))
#
#     # Draw the lane onto the warped blank image
#     cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
#     cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
#     out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
#
#     # plt.imshow(out_img)
#     # plt.plot(left_fitx, ploty, color='yellow')
#     # plt.plot(right_fitx, ploty, color='yellow')
#     # plt.xlim(0, 1280)
#     # plt.ylim(720, 0)
#
#     return {'img': out_img, 'ploty': ploty, 'too_few_points': len(left_lane_inds) < min_inds_per_lane or len(right_lane_inds) < min_inds_per_lane,
#             'r_center_out_of_view': False, 'l_center_out_of_view': False}


def convol(warped_bin, window_width=100, window_height=80, margin=100):
    """Find left and right lanes on warped image"""

    window_centroids = find_window_centroids(warped_bin, window_width, window_height, margin)
    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped_bin)
        r_points = np.zeros_like(warped_bin)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped_bin, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped_bin, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.dstack(
            (warped_bin, warped_bin, warped_bin)) * 255  # making the original road pixels 3 color channels
        output = cv2.addWeighted(template, 1.0, warpage, 0.5, 0.0)  # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped_bin, warped_bin, warped_bin)), np.uint8)

    return output


def find_window_centroids(image, window_width, window_height, margin):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # flags for stopping monitoring the lane centers when they go out of frame
    r_center_out_of_view = False
    l_center_out_of_view = False

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2

        if l_center is not None:  # left side center is still in the view
            l_min_index = int(max(l_center - margin + window_width, 0))
            l_max_index = int(min(l_center + margin, image.shape[1]))

            conv = np.array(conv_signal[l_min_index:l_max_index])
            if (conv > window_height).any():
                l_center = np.argmax(conv) + l_min_index - offset

            if l_center + offset >= image.shape[1] - 1 or l_min_index <= 0:
                if l_center_out_of_view:
                    l_center = None
                else:
                    l_center_out_of_view = True

        if r_center is not None:  # right side center is still in the view
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center - margin + window_width, 0))
            r_max_index = int(min(r_center + margin, image.shape[1]))

            conv = conv_signal[r_min_index:r_max_index]
            if (np.array(conv) > window_height).any():
                r_center = np.argmax(conv) + r_min_index - offset

            if r_center + offset >= image.shape[1] - 1 or r_min_index <= 0:
                if r_center_out_of_view:
                    r_center = None
                else:
                    r_center_out_of_view = True

        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)

    if center is None:
        return output

    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
                max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


