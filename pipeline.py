import cv2 as cv
from utils import *
from line import Line


class Pipeline:
    def __init__(self):
        # camera matrix used undistort the images
        camera_params = np.load("camera_calib_result.npz")
        self.mtx = camera_params['mtx']
        self.dist = camera_params['dist']

        self.frame_idx = 0  # video frame index

        # unwarping function ratios
        self.bot_width = .76  # percent of bottom trapezoid height
        self.mid_width = .08  # percent of middle trapezoid height
        self.hight_pct = .62  # percent for trapezoid height
        self.bott_trim = .935  # percent from top to bottom to avoid the car hood

        # left and right lane line coefficients
        self.left_line = None
        self.right_line = None

        # non-optimized sliding window search params
        # Set the width of the search windows to 2*margin for each frame level
        self.margin = 140
        # Set minimum number of points found to recenter window at each level.
        # If less points found, keep the previous center.
        self.minpix = 400

        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    def draw_lane(self, img):
        self.frame_idx += 1
        # Undistort the camera image
        undist_img = cv.undistort(img, self.mtx, self.dist, None, self.mtx)

        # Apply Sobel and color filters to extract the lines
        filtered_img = self._apply_filters(undist_img)

        # Unwarp image to bird-eye view for easier lane win analysis
        warpedImg, M, Minv = self.unwarp_image(filtered_img)
        warped_binary_img = np.zeros(warpedImg.shape[0:2], dtype=np.uint8)
        warped_binary_img[warpedImg[:, :, 0] > 50] = 1               # get rid of some of the noise pixels

        # Use sliding window to discover the lane lines coefficients (2nd deg polynomial) in the unwarped image
        win = self._sliding_win(warped_binary_img, margin=self.margin, minpix=self.minpix)
        self.left_line = win['left_line']
        self.right_line = win['right_line']
        # unwarped image with the lane lines colored and the search rectangles drawn
        bird_view_img = win['search_visualization_img']

        # Create an image palette to draw the lane polygon on
        warp_zero = np.zeros_like(warped_binary_img).astype(np.uint8)
        img_palette = np.dstack((warp_zero, warp_zero, warp_zero))

        # extract left and right line x_points at y_points
        y_points = np.linspace(0, warped_binary_img.shape[0] - 1, warped_binary_img.shape[0])
        left_line_x_points = self.left_line.get_points(y_points)
        right_line_x_points = self.right_line.get_points(y_points)

        # fill polygon between left and right lines on the image palette
        self._fill_polygon_on_palette(img_palette, left_line_x_points, right_line_x_points, y_points)

        # Warp the img_palette with the polygon back to the original image space using inverse perspective matrix (Minv)
        warped_polygon_layer_img = cv2.warpPerspective(img_palette, Minv, (undist_img.shape[1], undist_img.shape[0]))

        # Combine the result with the original image
        lane_img = cv2.addWeighted(undist_img, 1, warped_polygon_layer_img, 0.3, 0)

        # Calculate the radii of the curvature in meters
        y_eval = np.max(y_points)
        left_curverad = self.evaluate_lane_curve_radius(line=self.left_line, eval_pnt_y=y_eval)
        right_curverad = self.evaluate_lane_curve_radius(line=self.right_line, eval_pnt_y=y_eval)
        curve_radius = (left_curverad + right_curverad) // 2

        # calculate the distance the car is off center of the lane (meters)
        center_offset_m = self.calculate_dist_off_center(undist_img, warped_polygon_layer_img)

        return self.overlay_meta_image(lane_img, filtered_img, bird_view_img, frame_num=self.frame_idx,
                                       curve_rad=curve_radius, center_offset=center_offset_m)

    def _fill_polygon_on_palette(self, img_palette, left_line_x_points, right_line_x_points, y_points):
        """
        Draw a polygon on image palette

        :param img_palette: palette to draw the polygon on
        :param left_line_x_points: left line points
        :param right_line_x_points: right line points
        :param y_points:
        :return:
        """
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_line_x_points, y_points]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line_x_points, y_points])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the unwarped image palette
        cv2.fillPoly(img_palette, np.int_([pts]), (0, 255, 0))

    def calculate_dist_off_center(self, img, warped_lane_img):
        """
        Determine the distance car is off center by comparing the
        :param img: original undistorted image
        :param warped_lane_img: new warped lane image
        :return:
        """
        front_perspective_row = int(img.shape[0] * self.bott_trim)
        front_perspective_inds = np.array(warped_lane_img[front_perspective_row]).nonzero()[0]
        mid_lane_x = img.shape[1] // 2
        center_diff_pix = (front_perspective_inds[0] + front_perspective_inds[-1]) // 2 - mid_lane_x
        center_diff_m = center_diff_pix * self.xm_per_pix
        return center_diff_m

    def evaluate_lane_curve_radius(self, line=None, eval_pnt_y=0):
        """
        Evaluate lane line curve radius at a point

        :param line: quadratic Line
        :param eval_pnt_y: point at which the curve radius is determined
        :return:
        """
        a = line.quad_coeff * self.xm_per_pix / (self.ym_per_pix ** 2)
        b = line.linear_coeff * self.xm_per_pix / self.ym_per_pix
        curve_radius = ((1 + (2 * a * eval_pnt_y * self.ym_per_pix + b) ** 2) ** 1.5) / np.absolute(2 * a)

        return curve_radius

    def _apply_filters(self, undist_img):
        """
        Apply Sobel and color filters to extract the lines

        :param undist_img:  undistorted camera image to apply filters to
        :return:
        """
        sobel_binary_x = abs_sobel(undist_img, orient=(1, 0), sobel_kernel=5, thresh=(20, 100))
        sobel_binary_y = abs_sobel(undist_img, orient=(0, 1), sobel_kernel=5, thresh=(20, 100))
        color_binary_s = color_select(undist_img, thresh=(100, 255), color_conv_type=cv.COLOR_RGB2HLS, color_conv_indx=2)
        color_binary_v = color_select(undist_img, thresh=(50, 255), color_conv_type=cv.COLOR_RGB2HSV, color_conv_indx=2)

        bin_img = np.zeros_like(undist_img)
        bin_img[((sobel_binary_x == 1) & (sobel_binary_y == 1)) | ((color_binary_s == 1) & (color_binary_v == 1))] = (255, 255, 255)
        return bin_img

    def unwarp_image(self, img):
        """
        :param img:  image to be unwarped
        :return: unwarped image, unwarp matrix M, warp matrix (inverse of M)

        Warping function was borrowed from Volker van Aken.  His ratios work better than the ones I came up with.
        """

        src = np.float32([[img.shape[1] * (.5 - self.mid_width / 2), img.shape[0] * self.hight_pct],
                          [img.shape[1] * (.5 + self.mid_width / 2), img.shape[0] * self.hight_pct],
                          [img.shape[1] * (.5 + self.bot_width / 2), img.shape[0] * self.bott_trim],
                          [img.shape[1] * (.5 - self.bot_width / 2), img.shape[0] * self.bott_trim]])

        left_off_pct = 1 / 8  # part of left cut
        right_off_pct = 1 / 4  # part of right cut

        dst = np.float32([[img.shape[1] * left_off_pct, 0], [img.shape[1] * (1 - right_off_pct), 0],
                          [img.shape[1] * (1 - right_off_pct), img.shape[0]],
                          [img.shape[1] * left_off_pct, img.shape[0]]])

        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

        return warped, M, M_inv

    def _sliding_win(self, binary_unwarped, margin=100, minpix=50):
        """
        Sliding window algorithm to discover lanes
        :param binary_unwarped: unwarped bird-view image
        :param margin: +/- margin to search pixels around the current center of the lane
        :param minpix: minimum number of pixels at window level that is required to count towards the lane.
        This helps eliminate some noise.
        :return: left and write line along with the bird-view image with the lane pixels highlighted
        """
        # Choose the number of sliding windows
        nwindows = 9

        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_unwarped[binary_unwarped.shape[0] // 2:, :], axis=0)

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_unwarped, binary_unwarped, binary_unwarped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Set height of windows
        window_height = np.int(binary_unwarped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_unwarped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one level
        for window_lvl in range(nwindows):
            # Identify window boundaries in y dir
            win_y_low = binary_unwarped.shape[0] - (window_lvl + 1) * window_height
            win_y_high = binary_unwarped.shape[0] - window_lvl * window_height

            # LEFT LANE
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

            # RIGHT LANE
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            right_lane_inds.append(good_right_inds)

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
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Color the pixels identified as parts of the lane line
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        return {'search_visualization_img': out_img,
                'left_line': Line(scalar=left_fit[2], linear_coeff=left_fit[1], quad_coeff=left_fit[0]),
                'right_line': Line(scalar=right_fit[2], linear_coeff=right_fit[1], quad_coeff=right_fit[0])
                }

    def overlay_meta_image(self, lane_img, filtered_img, bird_view_img, curve_rad=0, center_offset=0, frame_num=0):
        """
        Prepare the final pretty pretty output blend, given all intermediate pipeline images
        :param lane_img: color image of lane blend onto the road
        :param filtered_img: image after applying Sobel and color filters
        :param bird_view_img: bird's eye view with sliding window boxes drawn
        :param center_offset: differens between the middle of the image, the car position to the middle of both lanes
        :return: The frame image overlayed with the meta information
        :param frame_num: number of video frame

         Borrowed and modified from https://github.com/ndrplz/self-driving-car/blob/master/project_4_advanced_lane_finding/main.py
        """
        h, w = lane_img.shape[:2]
        thumb_ratio = 0.2
        thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)
        off_x, off_y = 20, 15
        # add a gray rectangle to highlight the upper area
        mask = lane_img.copy()
        mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h + 2 * off_y), color=(0, 0, 0), thickness=cv2.FILLED)
        lane_img = cv2.addWeighted(src1=mask, alpha=0.2, src2=lane_img, beta=0.8, gamma=0)
        # add thumbnail of binary image
        thumb_pre = cv2.resize(filtered_img, dsize=(thumb_w, thumb_h))

        lane_img[off_y:thumb_h + off_y, off_x:off_x + thumb_w, :] = thumb_pre
        # add thumbnail of bird's eye view (lane-line highlighted)
        thumb_img_fit = cv2.resize(bird_view_img, dsize=(thumb_w, thumb_h))
        lane_img[off_y:thumb_h + off_y, lane_img.shape[1] - off_x - thumb_w:lane_img.shape[1] - off_x, :] = thumb_img_fit

        if curve_rad > 1999:
            curve_rad = 'sraight'
        else:
            curve_rad = str(round(curve_rad, -2)) + 'm'

        # add text (curvature and offset info) on the upper right of the blend
        side_pos = ' left'
        if center_offset <= 0:
            side_pos = ' right'

        center = str(round(center_offset, 2)) + 'm' + side_pos

        cv2.putText(lane_img, 'Radius of Curvature = ' + curve_rad, (300, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(lane_img, 'Vehicle is ' + center + ' off center.', (300, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(lane_img, 'Frame:   ' + str(frame_num), (300, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                    cv2.LINE_AA)
        return lane_img





