import time
import numpy as np

# print(f'running scratch....')
# start = time.time()
# time.sleep(10)

# end = time.time() + 60

# print(f'end: {end}, start: {start}')
# mins = (end - start) // 60
# secs = int((end - start) % 60)

# print(f'elapsed time: {mins} m {secs} s')

# test_second_frame = list(range(30))

import re        
# text = " um my parents are from here um"

# text = re.sub(r'\bum\b', ' ', text)
# text = text.strip()

# goal: remove ums
# text = text.replace(' um', ' ')
# text = text.replace('um ', ' ')
# text = text.replace(' um ', ' ')
# text = text.strip()
# print(f'processed text: \n[{text}]')

# text = "they ain't doing anything"
utterances = [
    "they ain't doing anything",
    "isn't the weather fantastic?",
    " um I like dogs um where are you going um"
]
text = utterances[2]

""" Replaces contractions & removes 'um's
"""
def remove_informalisms(text: str) -> str:
    contractions_dict = {
        "isn't" : "is not",
        "ain't" : "are not",
        "um"    : ""
    }
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in contractions_dict.keys()) + r')\b')
    text = pattern.sub(lambda x: contractions_dict[x.group()], text) # substitute informalisms
    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces left behind from removing 'ums'
    return text

text = remove_informalisms(text)
print(f'processed text: \n[{text}]')
