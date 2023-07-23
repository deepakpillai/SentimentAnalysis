import pandas as pd

class ExtractData():
    def __init__(self):
        pass

    def load_data(self, name, train=True):
        data = self.prepare_dataset(name, train)
        return data

    def prepare_dataset(self, name, train=True):
        train_data = pd.read_csv(name, encoding='unicode_escape', na_values='NaN')
        train_df = pd.DataFrame(train_data)
        train_df_dataset = train_df[['text', 'sentiment']]
        train_df_dataset = train_df_dataset.replace("negative", 0)
        train_df_dataset = train_df_dataset.replace("neutral", 1)
        train_df_dataset = train_df_dataset.replace("positive", 2)
        train_df_dataset.dropna(subset=['text'], inplace=True)
        train_df_dataset = train_df_dataset.drop(0)
        if train == False:
            train_df_dataset.drop(train_df_dataset.tail(43).index, inplace=True)
        print(f"\n\n\nTotal length of dataset: {len(train_df_dataset)}")
        return train_df_dataset