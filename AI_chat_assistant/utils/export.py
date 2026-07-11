import json


def export_chat(history):

    return json.dumps(history, indent=4)