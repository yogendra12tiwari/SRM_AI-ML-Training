import time


def stream_text(text):

    words = text.split()

    for word in words:
        yield word + " "
        time.sleep(0.03)