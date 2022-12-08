from pygame import mixer

mixer.init()
mixer.music.load("song1.wav")
mixer.music.set_volume(0.5)
mixer.music.play()

while True:
    print('press p to pause and r to resume')

    ch = input()
    if ch == 'p':
        mixer.music.pause()
    elif ch == 'r':
        mixer.music.play()
    
