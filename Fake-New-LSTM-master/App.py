import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.keras.layers import Embedding
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot


voc_size=5000
from tensorflow.keras.models import load_model
model1 = load_model('model.h5')
ps = PorterStemmer()
def analyze(headline):
    headline=headline.lower()
    headline=headline.split()
    headline = [ps.stem(word) for word in headline if not word in stopwords.words('english')]
    headline = ' '.join(headline)
    onehot_repr = [one_hot(words, voc_size) for words in headline]
    sent_length = 20
    embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
    x=np.array(embedded_docs)
    pred=model1.predict_classes(x)
    return(pred)



import streamlit as st
st.header('This is streamlit app')
text_1=st.text_area('enter the headline')
if st.button(label='analyze'):
    result=analyze(text_1)
    st.text(result)
    if result<=0.5:
        st.text(text_1+'have high chances of being fake')
    else:
        st.text(text_1+'is most probably a real headline')



