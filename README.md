# Airline Sentiment Prediction using BERT

API Endpoint: https://aviralv-airline-sentiment-bert.hf.space/run/predict

API Documentation: https://aviralv-airline-sentiment-bert.hf.space/?view=api

### Approach
First I analysed the data and I found that there was a huge imbalance in the dataset, to resolve this I used Textattack for augumentation of data.
Before the augumenting the dataset I used the following techniques to clean the data & reduce the noise:
- Removed the @usernames
- Removed the URLs
- Removed hashtags
- Replacement of emojis with their meaning

After cleaning the data I used EasyDataAugment of Textattack to augment the data, augmenting the data helped me to increase the accuracy of the model by more than 3%. I also tried using Clare(It replaces the words with their synonyms) but that was very resource intensive & it was taking very long to get output.

### Model
Since, this was a binary classification task I used BERT for training the model. I used the pretrained BERT model from Huggingface transformers library. I used the BERT model with the following parameters:
- BERT-base-uncased
- Max length of the input sequence: 128
- Learning rate: 3e-5
- Batch size: 32

### Results
The dataset was split into 80:20 ratio for training & validation.
I got the following results after training the model:
Training loss: 0.0137
Validation loss: 0.1209
Training accuracy: 0.9955
Validation accuracy: 0.9794

## How to run the code
The given code is trained on custom dataset provided with this repo, if you want to train it on your own dataset then you can do so by referring to train.ipynb file.

To create an API endpoint at localhost, run the following command:
```uvicorn api:app --reload```

After running the above command, you can access the API documentation at http://127.0.0.1:8000/docs