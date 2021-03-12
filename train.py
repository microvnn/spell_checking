import itertools
from os import path, makedirs
from tools import *
from nltk import ngrams
from tqdm import tqdm
import numpy as np
import nltk
import glob

# Build the neural network
# this is adapted from the seq2seq architecture, which can be used for Machine Translation, Text Summarization Image Captioning ...
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, LSTM, Bidirectional
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import Adam

#print(re.__version__)
#print(nltk.__version__)
#print(keras.__version__)

# load_model(path.join("/content/drive/MyDrive/", "Spell_checking", 'models', "spell_04.h5"))
#exit()


# re.match(alphabet, p.lower()):
alphabet_filter = re.compile(alphabet_regex)
print(alphabet_filter.match("aaäaa"))
#exit()

def readlines(filename):
    f = open(filename, "r", encoding="utf-8")
    return [x.strip() for x in f.readlines()]

input_path = path.join("./datasets")
files = glob.glob(path.join(input_path, "*.*.txt"))
training_data = []
for f in files:
    print(f'read : {f}')
    training_data += [s.strip() for s in readlines(f)]
    # break

print(f'Total phrase: {len(training_data)}')
#print(training_data[0])
#print(not alphabet_filter.match(training_data[0].lower()))
#exit()

phrases = itertools.chain.from_iterable(extract_phrases(text) for text in training_data)
phrases = [p.strip() for p in phrases if len(p.split()) > 1]

# divide document into 5-grams
# a single Vietnamese word cant contain more than 7 characters ( nghiêng )
NGRAM = 5
MAXLEN = 40

def gen_ngrams(words, n=5):
    return ngrams(words.split(), n)

list_ngrams = []
for p in tqdm(phrases):
    if not alphabet_filter.match(p.lower()):        
        continue
    for ngr in gen_ngrams(p, NGRAM):
        if len(" ".join(ngr)) < MAXLEN:
            list_ngrams.append(" ".join(ngr))

del phrases
list_ngrams = list((list_ngrams))
print(len(list_ngrams))


# So a 5-grams contain at most 7*5 = 35 character (except one that has spell mistake)
# add "\x00" padding at the end of 5-grams in order to equal their length
def encoder_data(text, maxlen=MAXLEN):
    try:
      text = "\x00" + text
      x = np.zeros((maxlen, len(alphabet)))
      for i, c in enumerate(text[:maxlen]):          
          x[i, alphabet.index(c)] = 1
      if i < maxlen - 1:
          for j in range(i + 1, maxlen):
              x[j, 0] = 1
      return x
    except:
      print(text)

def decoder_data(x):
    x = x.argmax(axis=-1)
    return "".join(alphabet[i] for i in x)


print(encoder_data("Tôi tên là việt hoàng").shape)
print(decoder_data(encoder_data("Tôi tên là Việt Hoàng")))

encoder = LSTM(256, input_shape=(MAXLEN, len(alphabet)), return_sequences=True)
decoder = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))
model = Sequential()
model.add(encoder)
model.add(decoder)
model.add(TimeDistributed(Dense(256)))
model.add(Activation("relu"))
model.add(TimeDistributed(Dense(len(alphabet))))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"])

model.summary()
from keras.utils import plot_model

plot_model(model, to_file=path.join("/content/drive/MyDrive/", "Spell_checking", "model.png"))

from sklearn.model_selection import train_test_split

train_data, valid_data = train_test_split(list_ngrams, test_size=0.2, random_state=42)
del list_ngrams


BATCH_SIZE = 512


def generate_data(data, batch_size):
    cur_index = 0
    while True:
        x, y = [], []
        for i in range(batch_size):
            y.append(encoder_data(data[cur_index]))
            x.append(encoder_data(add_noise(data[cur_index], 0.94, 0.985)))
            cur_index += 1
            if cur_index > len(data) - 1:
                cur_index = 0
        yield np.array(x), np.array(y)


train_generator = generate_data(train_data, batch_size=BATCH_SIZE)
validation_generator = generate_data(valid_data, batch_size=BATCH_SIZE)

# train the model and save to the Model folder
checkpointer = ModelCheckpoint(
    filepath=path.join("./models", "spell_{epoch:02d}.h5"),
    save_best_only=True,
    verbose=1,
)
model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_data) // BATCH_SIZE,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=len(valid_data) // BATCH_SIZE,
    callbacks=[checkpointer],
)
# model.save("./models", "spell_vietnamese.h5")
# model.save_pickle("/content/drive/MyDrive/", "Spell_checking", 'models',"spell_vietnamese.pkl")