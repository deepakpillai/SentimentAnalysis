import ExtrateData as ed
import ConvertToTensor as et
from torch.utils.data.dataloader import DataLoader
import Train as t
import re
import pickle
from pathlib import Path

class Main():
    def __init__(self, csv_name, encoder, train=True):
        self.csv_name = csv_name
        self.train = train

        if train == True:
            self.data = []
            folder_path = Path('Data')
            folder_path.mkdir(parents=True, exist_ok=True)
            model_file_name = 'tensorfile.pkl'
            file_path = folder_path / model_file_name
            try:
                with (open(file_path, "rb")) as openfile:
                    self.data = pickle.load(openfile)
            except (OSError, IOError) as e:
                self.data = self.prepare_data(encoder)
                with open(file_path, 'wb') as f:
                    pickle.dump(self.data, f)
            print(f"Train file length: {len(self.data)}")
        else:
            self.data = []
            folder_path = Path('Data')
            folder_path.mkdir(parents=True, exist_ok=True)
            model_file_name = 'test_tensorfile.pkl'
            file_path = folder_path / model_file_name
            try:
                with (open(file_path, "rb")) as openfile:
                    self.data = pickle.load(openfile)
            except (OSError, IOError) as e:
                self.data = self.prepare_data(encoder)
                with open(file_path, 'wb') as f:
                    pickle.dump(self.data, f)

            print(f"Test file length: {len(self.data)}")

    def prepare_data(self, encoder):

        data = ed.ExtractData().load_data(self.csv_name, self.train)
        data_array = []
        for index, value in data.iterrows():
            text = str(value['text'])
            text = text.strip()
            text = self.clean_string(text)
            sentiment = value['sentiment']
            if text != '':
                data_tupple = ()
                embedding = encoder.get_tensor(text)
                data_tupple = data_tupple + (embedding, sentiment)
                data_array.append(data_tupple)

            # if index == 10:
            #     break

        return data_array

    def set_data_batch(self):
        batch_data = DataLoader(self.data, batch_size=30, shuffle=True)
        return batch_data

    def clean_string(self, string):
        text = re.sub('http://\S+|https://\S+', '', string) #for links
        return text


if __name__ == '__main__':
    encoder = et.ConvertToEmbed()
    train_batch = Main('text_classificatoin_train.csv', encoder, True).set_data_batch()
    test_batch = Main('text_classificatoin_test.csv', encoder, False).set_data_batch()
    t.ModelTraining().train_model(train_batch, test_batch)