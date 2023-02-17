#  构造Tokenizer
import joblib
from keras.preprocessing.text import Tokenizer

vocab = {"周杰伦", "陈奕迅", "电饭锅", "梵蒂冈", "阿松大"}

t = Tokenizer(num_words=None, char_level=False)

t.fit_on_texts(vocab)

for token in vocab:
    zero_list = [0] * len(vocab)
    token_index = t.texts_to_sequences([token])[0][0] - 1
    zero_list[token_index] = 1
    print(token, "的one-hot编码为", zero_list)

tokenizer_path = "./Tokenizer"
joblib.dump(t, tokenizer_path)

#  加载构造好的Tokenizer
