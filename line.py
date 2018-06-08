class Line():
    def __init__(self, scalar, linear_coeff, quad_coeff):
        # line coefficients
        self.scalar_coeff = scalar
        self.linear_coeff = linear_coeff
        self.quad_coeff = quad_coeff

    def get_points(self, data=()):
        return self.quad_coeff * data ** 2 + self.linear_coeff * data + self.scalar_coeff
