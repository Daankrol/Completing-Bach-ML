import numpy as np
# from sklearn import preprocessing


# read info from file
file_input = open("F.txt", 'r')
sounds = file_input.readlines()

voice_1 = []
voice_2 = []
voice_3 = []
voice_4 = []

# store each line of pitches as a list
for i in range(0,len(sounds)):
    voice_all = sounds[i].split()
    voice_1.append(int(voice_all[0]))
    voice_2.append(int(voice_all[1]))
    voice_3.append(int(voice_all[2]))
    voice_4.append(int(voice_all[3]))

### translate to pitch duration pairs
def reduce_to_pairs(dat):
    pairs = []
    i=0
    while i < len(dat):
        duration = 1
        while i+1 < len(dat) and dat[i+1] == dat[i]:
            duration += 1
            i += 1
        pairs.append([dat[i], duration])
        i += 1
    return pairs

pairs_4 = reduce_to_pairs(voice_4)

print(len(voice_4))
print(len(pairs_4))
print(pairs_4[0:len(pairs_4)])
