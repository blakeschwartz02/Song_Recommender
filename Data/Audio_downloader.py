'''
This file contains methods helpful for downloading
audio files from YouTube.
'''

from pytube import YouTube
from pytube import Playlist
from pydub import AudioSegment
import sys
import os

NUM_ARGS = 3
BASE_DIR = "Audios"


def download_video(url):
    '''
    Download video from youtube given the url. The video (mp4 format) is then
    converted to wav format. 

    Parameters:
        url (str): url of the video to be downloaded
    '''
    video = YouTube(url)
    video.streams.filter(only_audio=True).first().download()

    # convert mp4 to wav
    sound = AudioSegment.from_file(video.title + ".mp4", format="mp4")
    sound.export(url + ".wav", format="wav")
    os.remove(url + ".mp4")

def download_playlist(url):
    '''
    Download all videos from a playlist given the url. The videos (mp4 format) are then
    converted to wav format.

    Parameters:
        url (str): url of the playlist to be downloaded
    '''
    playlist = Playlist(url)

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

