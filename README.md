# SENTIMENT ANALYSIS WITH NATURAL LANGUAGE PROCESSING

We first split train.json dataset into training and test data. The splitting was done randomly with 80% of the dataset used as training and remaining 20% as test data. Then, each of them are preprocessed using using a data cleaning function that we implemented. The function includes: 
Removing NaN values
Checking and removing stopwords obtained by importing nltk stopwords file
Finding and removing the words ends in "n't" or is the word "not" or "no" or "never" using Regular Expression
Removing digits from 0-9 using Regular Expression for words that are not found in stopwords
After both training and test data are scrubbed, we vectorize them using CountVectorizer() (Appendix 1) and TfidfVectorizer() (Appendix 2) separately, resulting into 2 vectorized training data and 2 vectorized test data. To ensure the labels are formatted according to model requirements, we used LabelEncoder() (Appendix 3) to reformat them.

There is also Word Embeddings using Word2Vec that is more context sensitive and can extract more insight of the data. However, implementing the word embedding model itself requires another neural network which need to be trained and parameterized. Due to time constraint, we considered CountVectorizer and TF-IDF Vectorizer sufficient to fulfill our needs of vectorizing the reviews data.

We choose Recurrent Neural Network (RNN) to implement our classification model since it is better suited to analyzing temporal, sequential data, such as text or videos. It is also computationally more efficient and faster as compared to pretrained model such as BERT that requires large corpus and a lot of weights to update. We built the RNN model using TensorFlow’s Keras Sequential Library.

We implemented a hidden layer with output of 450 units (450 neurons) since it has suitable complexity with the dataset. When we tried adding more hidden layers and more output units, the model became too complex and overfit the training data. On the other hand, smaller output units compensate the model performance. The hidden layer utilized reLU as the activation function. It is the most commonly used activation function. We choose it for its simplicity and quick convergence. The function returns 0 if it receives any negative input, but for any positive value x it returns that value back. The input dimension is the number of distinct words in our cleaned training dataset that is fitted into Vectorizer2 (implemented usinf TfidfVectorizer). We added L1 and L2 regularizations as well to minimize overfitting likelihood.
 Lastly, the output layer was added with one output unit and sigmoid activation function. We use sigmoid since our model is a binary classifier in which the output is interpreted as a class label depending on the probability value of input returned by the function. Sigmoid function converts its input into a probability value between 0 and 1. We converted these values later on with a threshold of 0.5 (above or equal to 0.5 classified as 1 and the rest are 0). The loss function we used was binary_crossentropy since our labels is a binary (negative or positive sentiments). This loss function computes the cross-entropy loss between true labels and predicted labels. 
 
The hyperparameters set are:
- Number of epochs = 60
- Batch size = 32
- Validation data = 15% of the training data

After training the model, we evaluate it with the test data and obtained 91.02% accuracy. Further evaluation gave us the confusion matrix in Figure 3. From it, we can derive the following evaluation metrics:
- Specificity, measure of how well classifier can recognize the negative samples: 0.5231 
- Sensitivity, measure of how well classifier can recognize the positive samples: 0.9762
Accuracy, as a function of sensitivity and specificity: 0.9102
Precision, percentage of samples labeled as positive are actually such: 0.9230
Recall, percentage of positive samples are labeled as such: 12351265=0.9762
Lastly, the combination of Precision and Recall, F measure: F=2 x Precision x RecallPrecision + Recall=0.9489 

The learning rate chosen was 0.00003, the other optimizer parameters are default values of TensorFlow’s Adam. (See Appendix 3). Initially, we tried Adam’s default learning rate (0.0001) but it overfits easily since the vectorized data is rather too simple for our RNN with a hidden layer of 100 neurons model. We slowly lowered it down and found that 0.00003 was the most optimal rate.
We tried 2 different RNNs, each for CountVectorizer data and TF-IDF vectorized data. The number of epochs for CountVectorizer data is 30 as when we examined the epoch vs loss graph, the validation and training curves start to diverge after 30. For TF-IDF vectorizer, the curves start to diverge after 60 epochs, thus we choose 60 epochs as optimal parameter. We also implemented L1 and L2 regularization (Appendix 5) with values of L1=10-5 and L2=10-4 to prevent overfitting.
