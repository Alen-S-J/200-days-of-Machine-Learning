# Machine Learning Basics
Machine learning is a subset of artificial intelligence (AI) that involves the development of algorithms and models that enable computers to learn and make predictions or decisions without being explicitly programmed. In traditional programming, human programmers write specific instructions for a computer to perform a task. In contrast, in machine learning, algorithms learn patterns and relationships from data to generalize and make predictions or take actions.

#### Basic Principles:
#### Data Collection and Preparation:

Gathering relevant data is the first step. This data should be cleaned, organized, and formatted appropriately to be fed into a machine learning model.

#### Training Data and Labels:

Data is divided into a training set and a testing set. The training set contains examples with associated labels (or outputs), which serve as the ground truth for the model to learn from.

#### Model Selection:

Choosing an appropriate model or algorithm is crucial. Common models include decision trees, neural networks, support vector machines, and more. The choice often depends on the type of problem being solved (e.g., classification, regression, clustering).

#### Training the Model:

The selected model is fed the training data, and it adjusts its internal parameters to learn the patterns and relationships between the input data and the corresponding labels.

#### Evaluation and Validation:

The model's performance is evaluated using a separate testing set, which the model hasn't seen during training. Common metrics include accuracy, precision, recall, and F1 score.


#### Model Tuning and Optimization:

Based on the evaluation results, the model may be fine-tuned by adjusting hyperparameters or modifying the model's architecture to improve its performance.


### Deployment:

Once the model is satisfactorily trained and validated, it can be deployed to make predictions or decisions on new, unseen data.


#### Continuous Monitoring and Improvement:

Models need to be continuously monitored in real-world applications, and updates or improvements may be necessary as new data becomes available or as the model's performance degrades over time.

## Examples of  Machine learning
#### 1. Product recommendations
Product recommendations are one of the most popular applications of machine learning, as it is featured on most e-commerce websites. Using machine learning models, websites track your behaviour based on your browsing patterns, previous purchases, and shopping cart history.

Companies like Spotify and Netflix use similar machine learning algorithms to recommend music or TV shows based on your previous listening and viewing history. Over time, these algorithms understand your preferences to recommend new artists or films you may enjoy.

#### 2. Social media connections
Another example is the “people you may know” feature on social media platforms like LinkedIn, Instagram, Facebook, and Twitter. Based on your contacts or people you’ve previously followed, the algorithm suggests familiar faces from your real-life network that you might want to connect with and keep tabs on.

#### 3. Image recognition
Image recognition is another example of machine learning that appears in our day-to-day life. With the use of machine learning, programs can identify an object or person in an image based on the intensity of the pixels. This type of facial recognition is used in law enforcement. By filtering through a database of people to identify commonalities and matching them to faces, police officers and investigators can narrow down a list of crime suspects. 

#### 4. Speech recognition
Just like machine learning can recognize images, it can also translate speech into text. Software applications coded with AI can convert recorded and live speech into text files.

Voice-based technologies can be used in medical applications, such as helping doctors extract important medical terminology from a conversation with a patient. While this tool isn't advanced enough to make trustworthy clinical decisions, other speech recognition services provide patients with reminders to “take their medication” as if they have a home health aide by their side.

#### 5. Virtual personal assistants
Virtual personal assistants are devices you might have in your own homes, such as Amazon’s Alexa, Google Home, or the Apple iPhone’s Siri. These devices use a combination of speech recognition technology and machine learning to capture data on what you're requesting and how often the device is accurate in its delivery. They detect when you start speaking, what you’re saying, and deliver on the command. For example, when you say, “Siri, what is the weather like today?”, Siri searches the web for weather forecasts in your location and provides detailed information.

#### 6. Stock market predictions
Predictive analytics is a common type of machine learning that is applicable to industries as wide-ranging as finance, real estate, and product development. Machine learning classifies data into groups and then defines them with rules set by data analysts. After classification, analysts can calculate the probability of an action.

This type of machine learning can predict how the stock market will perform based on year-to-year analysis. Using predictive analytics machine learning models, analysts can predict the stock price for 2025 and beyond.

#### 7.Credit card fraud detection
Predictive analytics can help determine whether a credit card transaction is fraudulent or legitimate. Fraud examiners use AI and machine learning to monitor variables involved in past fraud events. They use this data to measure the likelihood that a specific event was fraudulent activity.

#### 8.Traffic predictions
When you use Google Maps to map your commute to work or a new restaurant in town, it provides an estimated time of arrival. Google uses machine learning to build models of how long trips will take based on historical traffic data (gleaned from satellites). It then takes that data based on your current trip and traffic levels to predict the best route according to these factors.

#### 9.Self-driving car technology
A frequently used type of machine learning is reinforcement learning, which is used to power self-driving car technology. Self-driving vehicle company Waymo uses machine learning sensors to collect data of the car's surrounding environment in real time. This data helps guide the car's response in different situations, whether it is a human crossing the street, a red light, or another car on the highway.


# Supervised Learning

Supervised learning is a fundamental approach in machine learning, where the model is trained on a dataset that is already labelled. Each data point in the training dataset consists of input features (also called attributes or predictors) and corresponding labels or target values. The primary objective of supervised learning is to generalize from this labelled data to make accurate predictions or decisions when presented with unseen or new data.

### How Supervised Learning Works?

In supervised learning, models are trained using labeled datasets where the model learns about each type of data. Once the training process is completed, the model is tested based on the test data (a subset of the training set), and then it predicts the output.

### Example and Diagram:

![Supervised Learning](https://static.javatpoint.com/tutorial/machine-learning/images/supervised-machine-learning.png)


#### Example: Supervised Machine Learning

Suppose we have a dataset of different types of shapes which include square, rectangle, triangle, and polygon. The first step is to train the model for each shape.

- If the given shape has four sides, and all the sides are equal, then it will be labeled as a Square.
- If the given shape has three sides, then it will be labeled as a triangle.
- If the given shape has six equal sides, then it will be labeled as a hexagon.

After training, we test our model using the test set, and the model's task is to identify the shape.

The machine is already trained on all types of shapes, and when it finds a new shape, it classifies the shape based on the number of sides and predicts the output.


## Supervised Learning

Supervised learning involves training a model on a labeled dataset, where each data point has both features (inputs) and corresponding labels (outputs). The model learns to map the features to the correct labels by finding patterns and relationships within the data. The primary goal is to approximate or learn the mapping function so that, given new, unseen data, the model can accurately predict the labels.

### Key Characteristics of Supervised Learning:

- **Labeled Data:** Training data has predefined labels or outcomes.
- **Training Process:** The model iteratively adjusts its parameters based on the difference between predicted and actual labels.
- **Examples:** Classification (e.g., spam detection, image recognition) and regression (e.g., predicting house prices).

### Steps Involved in Supervised Learning:

1. Determine the type of training dataset.
2. Collect/gather the labeled training data.
3. Split the training dataset into training dataset, test dataset, and validation dataset.
4. Determine the input features of the training dataset, which should have enough knowledge so that the model can accurately predict the output.
5. Determine the suitable algorithm for the model, such as decision trees, random forests, logistic regression, support vector machines, or k-nearest neighbors.
6. Execute the algorithm on the training dataset. Sometimes we need validation sets as control parameters, which are subsets of training datasets.
7. Evaluate the accuracy of the model by providing the test set. If the model predicts the correct output, it means our model is accurate.

### Types of Supervised Learning Algorithms:

#### Regression Algorithms

Regression algorithms are used to predict a continuous numerical value, such as a house's price or a day's temperature. Different types of regression algorithms exist, such as:

- Linear Regression
- Polynomial Regression
- Lasso Regression
- Ridge Regression

#### Classification Algorithms

Classification algorithms are used to predict a categorical or discrete value, such as whether an email is spam. Some examples of classification algorithms include:

- Decision Trees
- Support Vector Machines (SVM)
- k-Nearest Neighbours (kNN)

### Popular Supervised Learning Algorithms:

- **Decision Trees**
- **Random Forests**
- **Logistic Regression**
- **Support Vector Machines (SVM)**
- **k-Nearest Neighbours (kNN)**
- **XGBoost Algorithm**
- **CatBoost Algorithm**
- **Naive Bayes Classifier**

### Supervised Learning Example:

#### Linear Regression

Linear regression is used to identify the linear relationship between a dependent variable and one or more independent variables. It works best with an easily interpretable dataset and is commonly used to predict the value of a continuous variable from the input given by multiple features or to classify observations into categories by considering their correlation with reference variables.


#### Logistic Regression

Logistic regression is a type of predictive modelling algorithm used for classification tasks. It estimates the probability of an event occurring based on the values of one or more predictor variables. It's easy to interpret and compare the results of this technique, making it suitable for binary classification problems.

#### Support Vector Machines (SVM)

Support Vector Machines (SVM) are robust algorithms that use a kernel to map data into a high-dimensional space and then draw a linear boundary between the distinct classes. They are often used for text classification or image recognition due to their accuracy in predicting categorical variables from large datasets.

#### Decision Trees

Decision Trees are popular supervised learning algorithms for classification and regression tasks using a "tree" structure to represent decisions and their associated outcomes. They are great for prediction problems because they can quickly determine which category a piece of data belongs to based on previous data training samples.

#### Random Forests

Random Forests are essentially multiple decision trees combined to form one powerful "forest" model with better predictive accuracy than individual trees. They work by randomly choosing sets of observations and variables at each node split within the forest, comparing several predictive models before coming up with an optimal outcome that maximizes accuracy.

#### k-Nearest Neighbors (kNN)

k-Nearest Neighbors (kNN) is a simpler supervised learning algorithm that models the relationship between a given data point and its “nearest” neighbors. It aims to classify and predict based on a certain number of ‘nearest neighbors’ having similar features or properties, either in the same class or otherwise. This makes it particularly suitable when there is no clear knowledge boundary, like with non-linear problems and complex datasets.

#### Naive Bayes Classifier

Naive Bayes Classification is a powerful machine learning technique used for both supervised and unsupervised learning. It uses Bayes' Theorem to calculate the probability that a given data point belongs to one class or another based on its input features. It is useful for various applications such as text classification, recommendation systems, sentiment analysis, and image recognition.

#### XGBoost Algorithm

XGBoost stands for “eXtreme Gradient Boosting” and is a robust machine-learning algorithm that can help you better understand your data and make informed decisions. It implements gradient-boosting decision trees, wherein it uses multiple trees to output the final result. It offers advantages in execution speed and model performance, making it popular in various machine learning tasks.

#### CatBoost Algorithm

CatBoost is a gradient boosting algorithm for decision trees developed by Yandex. It has applications in many industries and is used for search, recommendation systems, personal assistants, self-driving cars, and weather forecasting tasks. CatBoost works by building successive decision trees, trained with a reduced loss compared to the previous trees, and offers unique features that set it apart from other boosting frameworks.

### How to Improve Supervised Learning Model Performance:

- Evaluate the model's accuracy using a holdout test data set.
- Use hyperparameter optimization to improve the model.
- Experiment with various data preprocessing techniques, such as oversampling and undersampling, to improve accuracy and avoid bias.
- Fine-tune your machine learning model by exploring different approaches and optimizing its performance for your specific use case.

### Building an End-to-End Machine Learning Solution with Supervised Learning:

Once the model has been trained, tested, and optimized, it can be deployed as an end-to-end machine learning solution by integrating it into a software application. Monitoring and optimizing its functioning over time are essential to ensure its performance meets expectations.

### Practical Tips for Developing Supervised Machine Learning Solutions:

- Use reliable data for training and testing the model.
- Choose the best model for the problem at hand based on the dataset's characteristics.
- Perform extensive tests on supervised models before deployment to ensure their performance as expected.

### Conclusion:

Supervised learning is a powerful approach to machine learning that involves training a model on labeled data to accurately predict new, unseen data. Various algorithms can be used in this process, and the choice depends on the problem at hand and the dataset's characteristics.


