import numpy as np


class Line():
    """
    Quadratic line class
    """
    def __init__(self, scalar, linear_coeff, quad_coeff):
        # line coefficients
        self.scalar_coeff = scalar
        self.linear_coeff = linear_coeff
        self.quad_coeff = quad_coeff

    def get_points(self, data=()):
        return self.quad_coeff * data ** 2 + self.linear_coeff * data + self.scalar_coeff

    def evaluate_curve_radius(self, xm_per_pix, ym_per_pix, eval_pnt_y=0):
        """
        Evaluate lane line curve radius at a point

        :param eval_pnt_y: point at which the curve radius is determined
        :return:
        """
        a = self.quad_coeff * xm_per_pix / (ym_per_pix ** 2)
        b = self.linear_coeff * xm_per_pix / ym_per_pix
        curve_radius = ((1 + (2 * a * eval_pnt_y * ym_per_pix + b) ** 2) ** 1.5) / np.absolute(2 * a)

        return curve_radius