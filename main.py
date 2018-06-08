from pipeline import Pipeline

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

"""
Process movies
"""

pipeline = Pipeline()

def process_image(image):
    """Process movie image"""
    return pipeline.draw_lane(image)


def process_movie(movie, movie_out):
    """
    Process movie image
    :param movie: movie
    :param movie_out: movie output
    :return:
    """
    clip = VideoFileClip(movie)
    mov_clip = clip.fl_image(process_image)  # NOTE: this function expects color images!!
    mov_clip.write_videofile(movie_out, audio=False)

process_movie('project_video.mp4', 'project_video_out.mp4')
