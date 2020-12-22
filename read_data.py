def get_voice(voice_number):
    """
    Get voice
    :param voice_number:
    :return:
    """
    file_input = open("F.txt", 'r')
    sounds = file_input.readlines()

    voice = []
    for i in range(0, len(sounds)):
        voice_all = sounds[i].split()
        voice.append(int(voice_all[voice_number]))

    return voice


def get_pairs(voice_number=1, method='cumulative'):
    """
    Translate to additive pitch duration pairs
    :param method: Cumulative or Shift
    :return:
    :param voice_number:
    :return:
    """
    voice = get_voice(voice_number=voice_number)
    pairs = []
    i=0
    while i < len(voice):
        duration = 1
        while i+1 < len(voice) and voice[i+1] == voice[i]:
            if method == 'shift':
                pairs.append([voice[i], duration])
            duration += 1
            i += 1
        pairs.append([voice[i], duration])
        i += 1
    return pairs

