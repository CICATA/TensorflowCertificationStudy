import json
import os
import wget
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 3"

# Load sarcasm dataset
data_dir = 'sarcasm'
filename = os.path.join(data_dir, 'sarcasm.json')
if not os.path.exists(data_dir):
    # Download dataset
    os.mkdir(data_dir)
    url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json'
    filename = wget.download(url, out=data_dir)

with open(filename, 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
print(len(word_index))
print(word_index)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)


# End of file tag
print('eof')
