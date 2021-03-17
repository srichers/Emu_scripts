import moviepy.editor as mp

directory = "/ocean/projects/phy200048p/shared/plots/fiducial"
clip = mp.VideoFileClip(directory+"/"+"2D_movie_fiducial.gif")
clip.write_videofile(directory+"/"+"2D_movie_fiducial.mp4")
