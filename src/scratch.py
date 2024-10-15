import time
import numpy as np


text = ["I am going to the mall", "will you come with me"]
# required: pip install absl-py tensorflow


""" returns a (512, len(utterances)) np array where 512 is the shape of
    the embedding of each utterance, and len(utterances) means that there
    is a row/embedding for every utternace
"""
def get_sentence_embedding(utterances: list[str]) -> np.ndarray:
    print(f'getting sentence embedding...')
    from absl import logging
    import tensorflow_hub as hub

    logging.set_verbosity(logging.ERROR)
    print(f'loaded imports...')

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"    
    print(f'loading model now...')
    model = hub.load(module_url)
    print(F'model loading complete, getting embedding now...')
    embedding = model(utterances) # run inference on the text, which expects a list of msgs

    for i, message_embedding in enumerate(np.array(embedding).tolist()):
        print("Message: {}".format(utterances[i]))
        # print("Embedding size: {}".format(len(message_embedding)))
        # message_embedding_snippet = ", ".join(
        #     (str(x) for x in message_embedding[:3]))
        # print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

    embedding = embedding.numpy() # cvt to np array and reshape for dimensions (512,)
    embedding = np.squeeze(embedding.T)

    print(f'final embedding shape with {len(utterances)} items in list: {embedding.shape}\n')

    return embedding

embedding = get_sentence_embedding(text)