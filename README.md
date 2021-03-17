# Predicting Housing Prices
![alt text](http://blog.time2move.ca/files/2014/09/Rising-house-prices.jpg)
## What is this repository?
As an upcoming ML engineer, I challenged myself to put my machine learning skills to the test. I challenged myself by tackling the [Housing Prices Challenge on Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The goal of this challenge is to predict the prices of houses in Ames, Iowa based on a given set of features. To be exact, there are 79 features in total. This project allows the engineer (in this case myself) to practice critical Data Science & Machine Learning techniques.    
    
This repository is organized via 4 folders: Submissions, Data, Code, and Models. In the Submissions folder, you will see the various submissions that I have made. In the Data folder, you will find the necessary datasets as well as any other necessary information. In the Code folder, you will see my various Jupyter Notebooks & Python Scripts for the project. Finally, in the models folder, you will see the various models that I have saved.  
    
The model is evaluated using the Root Mean Square Error, as this is the metric we are trying to minimize. My best model has a RMSE of __0.13757__. This currently ranks in the __top 43%__. In reality, my solution would be much higher because 1)some solutions have an unfeasible RMSE of 0.0 and 2)the solutions that have a RMSE of 0.00044 incorporated data from other sources besides the provided data. My best model is my tuned CatBoost Model.
  
_Note:_ you may use my solution as a reference; however, I would strongly advise you to tackle this challenge on your own. The only way you will get better at machine learning is to practice it on your own. I do not condone nor am I responsible for any cheating that may occur as a result of this repository.
## Machine Learning Project Checklist:
This checklist is what I use for every ML project. This goes through every major step & ensures that I have done everything correctly.  
1. Framing the Problem - Complete
2. Getting the Data - Complete
3. Exploring the Data - Complete
4. Data Preprocessing - Complete
5. Model Development - Complete
6. Model Tuning/Ensemble Learning - Complete
7. Deploying Model on Test Set & Presentation of Solution - Complete
## What tools are used in this project?
* [Pandas](https://pandas.pydata.org/docs/getting_started/index.html)
* [Numpy](https://numpy.org)
* [Matplotlib](https://matplotlib.org)
* [Seaborn](https://seaborn.pydata.org)
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [Unittest](https://docs.python.org/3/library/unittest.html)
* [Sys](https://docs.python.org/3/library/sys.html)
* [CatBoost](https://catboost.ai)
## References
* [Hands-On Machine Learning with Scikit-Learn, Keras, and Tensorflow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646)
* [Medium Article about Unit Testing in Python](https://medium.com/techtofreedom/unit-testing-in-python-23b129add2b)
* [Medium Article about sys module usuage](https://dkhambu.medium.com/importing-files-in-python-repository-28ab49fade37)
## Future Adjustments
In reality, there are infinite adjustments I could make to improve my score; however, here a couple fruitful ones:
* Combine the Tuned-CatBoost model with some other models (Linear Regression & Support Vector Machines seem promising)
* Feature Engineering: I could maybe cut down the categories for certain features.
* Feature Importance: Further feature selection. Use my model to make better selections for features.
## Closing Remarks
This project was very enjoyable ,and I definitely learning a lot along the way! I would recommend this challenge to anyone who is looking to dive into Machine Learning & Data Science. It is quite simple, and the dataset is relatively small & not overwhelming. Overall, this challenge was really fun and a great learning experience!
## About the author
I am an undergraduate student @ Rutgers University New Brunswick, who is pursing bachelor degrees in Computer Science and Cognitive Science. Furthermore, I am pursing a certificate in Data Science. I have a passion for AI ,and I am always intriguied by its power. Feel free to contact me via [Linkedln](https://www.linkedin.com/in/jinalshah2002/).   
Enjoy!  
Jinal Shah
