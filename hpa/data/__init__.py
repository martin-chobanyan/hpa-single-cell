from .dataset import *


def parse_string_label(label_str):
    return [int(i) for i in label_str.split('|')]
