# import the necessary packages
from pytube import YouTube
from pytube import Playlist
from pydub import AudioSegment
import sys
import os

NUM_ARGS = 3
BASE_DIR = "Audios"

# Download the video
# YouTube('https://www.youtube.com/watch?v=kXYiU_JCYtU&list=PL6Lt9p1lIRZ311J9ZHuzkR5A3xesae2pk').streams.filter(only_audio=True).first().download()

# sound = AudioSegment.from_file("numb.mp4", format="mp4")
# sound.export("output_file.wav", format="wav")

# Download playlist
playlist = Playlist('https://www.youtube.com/watch?v=XB16AkSGLqk&list=PLxI9rM7N2E01D8pIt0sYF-k5JY4OfAJDe')

# create dir for playlist if not exists
if not os.path.exists(BASE_DIR + "/" + playlist.title):
    os.makedirs(BASE_DIR + "/" + playlist.title)

# change workind dir
os.chdir(BASE_DIR + "/" + playlist.title)

# print current working dir
print(os.getcwd())

for video in playlist.videos:
    try:
        video.streams.filter(only_audio=True).first().download()
    except (Exception):
        continue


# get all files in current dir
files = os.listdir()

# convert all files to wav
for file in files:
    sound = AudioSegment.from_file(file, format="mp4")
    sound.export(file[:-4] + ".wav", format="wav")
    os.remove(file)

