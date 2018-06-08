import matplotlib.image as mpimg
import numpy as np

from pipeline import Pipeline

p = Pipeline()

frames = np.arange(990, 1005)
for f in frames:
    fname = "vid/frame{}.jpg".format(f)
    img = mpimg.imread(fname)
    out_img = p.draw_lane(img)

    mpimg.imsave(fname.replace("vid/", "vid_test/out_"), out_img)
