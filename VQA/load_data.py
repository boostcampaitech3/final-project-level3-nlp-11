import pandas as pd
import json


class VQADataset():
    pass

class VQADataloader():
    pass


def load_dataset(data_path):

    with open('data_path', 'r') as f:
        json_data = json.load(f)

    df= pd.DataFrame(data_path)
    #img_df = df['image']
    #question_df = df.drop(labels= 'image', axis=1)
    #return img_df, question_df
    return df




