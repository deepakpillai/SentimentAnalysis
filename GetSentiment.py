import tensorflow_hub as hub
import torch
from pathlib import Path

class GetSentiment:
    def __init__(self):
        self.encoder = self.load_encoder()
        self.model = self.load_model()
        print("loaded")
    def load_encoder(self):
        module_url = "universal-sentence-encoder-large_5"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
        model = hub.load(module_url)
        print("module %s loaded" % module_url)
        return model

    def embed(self, input):
        return self.encoder(input)

    def get_tensor(self, input):
        input = [input]
        tf_tensor = self.embed(input)
        np_array = tf_tensor.numpy()
        pt_array = torch.from_numpy(np_array).to(torch.float32)
        return pt_array

    def load_model(self):
        model_path = Path('Model')
        model_path.mkdir(parents=True, exist_ok=True)
        model_file_name = 'linearRelu.pth'
        file_path = model_path / model_file_name
        return torch.load(file_path)

    def get_sentiment_for_string(self, string):
        # negative 0; neutral 1; positive 2
        sentiment = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        embedding = self.get_tensor(string)
        logits = self.model(embedding)
        pred = logits.argmax(1)
        return sentiment[pred.item()]



sentiment = GetSentiment().get_sentiment_for_string("Im going out")
print(sentiment)