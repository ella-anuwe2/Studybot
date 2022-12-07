from playsound import playsound
import glob
import multiprocessing

p = multiprocessing.Process(target=playsound, args=("song1.wav",))
def create_playlist(path):
    for song in glob.glob(path):
        print('playing...'+ song)
        playsound(song)

def play():
    create_playlist('*.wav')

play()