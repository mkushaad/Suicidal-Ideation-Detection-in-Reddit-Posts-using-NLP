Dataset from Kaggle: https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch


PROJECT REPORT:

Chapter 1 Introduction 

1.1	Abstract
Suicide is the fourth leading cause of death, with a person dying from suicide every 43 seconds resulting in 720,000 death every year[1]. Suicidal rates are high in teenagers and young adults. The widespread use of social media through the never-ending, ever-growing internet offers a new platform to study those who are at risk of suicide. We used Machine Learning, specifically Natural Language Processing, or NLP to detect suicidal ideation through posts shared on popular social media platform, Reddit. 

1.2 Introduction

Over 16 million people attempt suicide each year and about 720,000 of them die from suicide. Suicide is the 4th leading cause of death in age 14-29[1]. Since each day more people are starting to use the internet, and social media specifically, it has emerged as a new platform to provide data for those who are at risk of suicide and a means of possible early intervention. We used dataset from Kagle, consisting of posts from popular social media platform Reddit[1]. Preprocessing and Lemmatization of the data is done, thereafter vectorization is done using  TF-IDF vectorizer. Logistic Regression is implemented to train and test the machine learning model using ‘sklearn’ library. The dataset was split into training and validation subsets, in an 80:20 ratio, ensuring the model could generalize to unseen data. Metrics such as accuracy, precision, recall, and F1-score were calculated on the validation set to assess the model's performance. The chosen Logistic Regression model achieved high recall, ensuring fewer instances of suicidal ideation went undetected. The model was saved using pickle, a python library, to be used in the future and easy accessibility

Chapter 2 Methodology 

The classification of posts aims to find if the user has suicidal ideation or not. This is done through data preprocessing, lemmatization, vectorization, model training, and saving the model for future accessibility. (Figure 1 summarizes the processes)

![image](https://github.com/user-attachments/assets/277f1249-a8f5-4ae7-b56c-4d73b05caf54)

2.1 Data Collection
The dataset was taken from Kaggle.com. It is a collection of suicidal posts from the subreddits r/SuicideWatch and r/depression and non-suicidal posts from r/teenagers from the social media platform Reddit. These were gathered by using Pushshift API[2].

2.2 Data Preprocessing
In this step the data is cleaned from unnecessary information present within the data. This was undertaken by using ‘re’ python library where, firstly each null row from the dataset was removed. Thereafter, special characters (*, ?, !, @, etc.), emojis, numbers and any other character other than the alphabets were removed from each row. All the uppercase letters were converted to lowercases. Then ‘spacy’ python library was used to transform each word into its root or base word, through the process of Lemmatization. This is essential to reduce the number of vectors or tokens, making it more effective for training. After saving all this processed data, it was split into 80-20 for training and testing, respectively. 

2.3 Vectorization 
TF-IDF vectorizer is used to turn the processed dataset into vectors, or a sparse matrix of top 5000 words. We used the ‘max_feature()’ of ‘tfidfvectorizer’ because of the large dataset having high vocabulary diversity. 

2.4 Model Training and Testing
The data was split 80-20, for training and testing, respectively. Logistic Regression method is used to train the machine learning model. The python library ‘sklearn’ was used import the untrained model. After training the model was saved using the ‘pickle’ python library for future accessibility. Thereafter the model was testing on the testing dataset and evaluated.

2.5 Model Evaluation
Accuracy, Precision, F1 Score and Confusion matrix were evaluated for the model. 

Chapter 3 Result and Discussion

3.1 Result
The accuracy and precision of individual label (suicidal and non-suicidal) was calculated along with the overall accuracy of the trained and tested model. The overall accuracy of the model turned out to be 93.01%. 
The other evaluated quantities were as shown in figure 2.

![image](https://github.com/user-attachments/assets/57857660-1306-4afb-bb38-dca0f8a551c1)

Furthermore, the confusion matrix is given in figure 3.

![image](https://github.com/user-attachments/assets/58e22d8d-050b-4854-b76e-832bc95ffd2b)

The project is based on real-time predictions of the user data provided. Here is an example of real-time predictions by the model. (Figure 4)

![image](https://github.com/user-attachments/assets/ef6e1b8b-126a-4315-b7d0-4de9fd9162e6)

3.2 Discussion
The aim of the project is to detect suicidal ideation in Reddit posts, leading to early intervention and helping those in need. With accuracy of 93.01%, there is still need for improvement and more fine tuning is required to get more accuracy. This will lead to less manual labor to detect cases of suicide online, hence automating the process. The model itself is not perfect, and more data is required for an almost perfect model to be really used in a real-time environment. 

Chapter 4 Conclusion and Future Work 

The model is working with an accuracy of 93.01% however is weak towards straight statements like “live” or “I want to live”, which it declares ‘suicidal’. Work needs to be done regarding this, one way is to develop a more rigid and complex model. 
Future Work and Implementation
More complex models like transformer models and even neural network based deep learning models can be implemented to make a more robust and accurate model.
This project, after much refining, can be implemented on Reddit, to detect suicidal ideation on posts precisely, and efficiently. 
We can transform it into defining emotions, especially neutral depressions, which will make it work on a wide and varied data.
Since the model takes data in general, we can train it for Twitter (now X), Facebook and many other social media websites and implement the model on those websites making it a universal type of model.

References
[1] World Health Organization (WHO) [Online]. Report published on 29th Aug 2024. 
      Accessed on 14th Jan 2024:  https://www.who.int/news-room/fact-sheets/detail/suicide
      
[2] Dataset from Kaggle: https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch
