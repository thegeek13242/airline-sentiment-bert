import re
import emoji
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
model.load_weights('weights.h5')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocess the text before feeding it to the model
def preprocess(text):
    text = emoji.demojize(text)
    text = text.replace(":", " ")
    text = ' '.join(text.split())
    text = re.sub("@[A-Za-z0-9]+", "", text)
    text = re.sub("#", "", text)
    text = re.sub("https?://[A-Za-z0-9./]+", "", text)
    text = re.sub("[^a-zA-Z.!?']", " ", text)
    return text

# Predict the sentiment of the text
def predict(text):
    text = preprocess(text)
    tf_batch = tokenizer([text], max_length=128, padding=True,
                         truncation=True, return_tensors='tf')
    tf_outputs = model(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    labels = ['Negative', 'Positive']
    label = tf.argmax(tf_predictions, axis=1)
    label = label.numpy()
    return labels[label[0]]
