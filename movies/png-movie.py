import os
import imageio
import moviepy.editor as mp
#import ffmpeg
import glob

#directory where plot image files are located
directory_inputs = sorted(glob.glob("/ocean/projects/phy200048p/shared/plots/*/"))
for d in directory_inputs:
    print("currently checking directory: "+d)
    #taking off the last "/" so I can do path.split
    directory_short = d[:-1]
    #grabbing current directory name
    directory_current = os.path.split(directory_short)[-1]
    filename = "2D_movie_"+directory_current #**specify the filename here**
    #If the gif file already exists, don't need to make it again, and assume the mp4 file has also been created (since the mp4 is created from the gif)
    if os.path.exists(d+filename+".gif"):
        print("output directory already exists and contains gif and movie")
        continue
    #If directory doesnt have  gif in it, run the movie (gif + mp4) routine
    else:
        print("output directory doesn't have gif and mp4 movies in it yet, running movie script")
        writer = imageio.get_writer(d+filename+".gif", fps=10)
        for file in sorted(glob.glob(d+"plt*3x3subplot.png")): #**specify here which png files you want**
            writer.append_data(imageio.imread(file))
        writer.close()

       # clip = mp.VideoFileClip(d+filename+".gif")
       # clip.write_videofile(d+filename+".mp4")
