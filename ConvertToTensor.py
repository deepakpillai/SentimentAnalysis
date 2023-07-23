import tensorflow as tf
import tensorflow_hub as hub
import torch

class ConvertToEmbed():
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        module_url = "universal-sentence-encoder-large_5"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
        model = hub.load(module_url)
        print("module %s loaded" % module_url)
        return model

    def embed(self, input):
        return self.model(input)

    def get_tensor(self, input):
        input = [input]
        tf_tensor = self.embed(input)
        np_array = tf_tensor.numpy()
        pt_array = torch.from_numpy(np_array).to(torch.float32)
        return pt_array