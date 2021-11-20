import pandas as pd


PATH = 'C:/Users/LM/Downloads/dataset/test.csv'
TRAINING_PATH = 'C:/Users/LM/Downloads/dataset/train.csv'

Test_df = pd.read_csv(PATH)
Training_df = pd.read_csv(TRAINING_PATH, encoding='latin-1')

print(Training_df[Training_df['label']==1].shape)