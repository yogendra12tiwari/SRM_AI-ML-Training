import time


def stream_text(text: str):
    """
    Simulate streaming response word by word.
    """

    words = text.split()

    for word in words:
        yield word + " "
        time.sleep(0.02)