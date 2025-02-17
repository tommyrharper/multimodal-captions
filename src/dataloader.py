from datasets import load_dataset

def load_flickr30k():
    dataset = load_dataset("nlphuji/flickr30k")
    return dataset
