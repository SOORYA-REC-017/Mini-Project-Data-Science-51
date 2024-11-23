import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, \
    precision_recall_fscore_support, roc_auc_score
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, Conv2D
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
from imblearn.over_sampling import SMOTE
import json

# Set random seed for reproducibility
tf.random.set_seed(1234)

# Hyperparameters
epochs_number = 1  # number of epochs for the neural networks
test_set_size = 0.1  # percentage of the test size comparing to the whole dataset
oversampling_flag = 0  # set to 1 to over-sample the minority class
oversampling_percentage = 0.2  # percentage of the minority class after the oversampling comparing to majority class

def save_model_results(model_name, y_test, prediction):
    """Save model results and statistics to JSON files"""
    try:
        # Create result dictionary
        results_dict = {
            'model_name': model_name,
            'accuracy': float(100 * accuracy_score(y_test, prediction)),
            'rmse': float(mean_squared_error(y_test, prediction, squared=False)),
            'mae': float(mean_absolute_error(y_test, prediction)),
            'f1_score': float(100 * precision_recall_fscore_support(y_test, prediction)[2][1]),
            'auc': float(100 * roc_auc_score(y_test, prediction)),
            'confusion_matrix': confusion_matrix(y_test, prediction).tolist()
        }
        
        # Create stats dictionary
        stats = {
            'normal_consumers': int(y[y['FLAG'] == 0].count()[0]),
            'fraud_consumers': int(y[y['FLAG'] == 1].count()[0]),
            'total_consumers': int(y.shape[0]),
            'no_fraud_percentage': float(y[y['FLAG'] == 0].count()[0] / y.shape[0] * 100),
            'test_set_no_fraud': float(y_test[y_test == 0].count() / y_test.shape[0] * 100)
        }

        # Save to JSON files
        with open('model_results.json', 'w') as f:
            json.dump(results_dict, f)
        with open('stats.json', 'w') as f:
            json.dump(stats, f)

        # Print results for console output
        print(f"\nResults for {model_name}:")
        print(f"Accuracy: {results_dict['accuracy']:.2f}%")
        print(f"RMSE: {results_dict['rmse']:.4f}")
        print(f"MAE: {results_dict['mae']:.4f}")
        print(f"F1 Score: {results_dict['f1_score']:.2f}%")
        print(f"AUC: {results_dict['auc']:.2f}%")
        print("Confusion Matrix:")
        print(results_dict['confusion_matrix'])

    except Exception as e:
        print(f"Error saving model results: {e}")

def read_data():
    rawData = pd.read_csv('preprocessedR.csv')

    # Setting the target and dropping the unnecessary columns
    global y  # Make y global so save_model_results can access it
    y = rawData[['FLAG']]
    X = rawData.drop(['FLAG', 'CONS_NO'], axis=1)

    print('Normal Consumers:                    ', y[y['FLAG'] == 0].count()[0])
    print('Consumers with Fraud:                ', y[y['FLAG'] == 1].count()[0])
    print('Total Consumers:                     ', y.shape[0])
    print("Classification assuming no fraud:     %.2f" % (y[y['FLAG'] == 0].count()[0] / y.shape[0] * 100), "%")

    # columns reindexing according to dates
    X.columns = pd.to_datetime(X.columns)
    X = X.reindex(X.columns, axis=1)

    # Splitting the dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y['FLAG'], test_size=test_set_size, random_state=0)
    print("Test set assuming no fraud:           %.2f" % (y_test[y_test == 0].count() / y_test.shape[0] * 100), "%\n")

    # Oversampling of minority class to encounter the imbalanced learning
    if oversampling_flag == 1:
        over = SMOTE(sampling_strategy=oversampling_percentage, random_state=0)
        X_train, y_train = over.fit_resample(X_train, y_train)
        print("Oversampling statistics in training set: ")
        print('Normal Consumers:                    ', y_train[y_train == 0].count())
        print('Consumers with Fraud:                ', y_train[y_train == 1].count())
        print("Total Consumers                      ", X_train.shape[0])

    return X_train, X_test, y_train, y_test

def ANN(X_train, X_test, y_train, y_test):
    print('Artificial Neural Network:')
    model = Sequential()
    model.add(Dense(1000, input_dim=1034, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, validation_split=0, epochs=epochs_number, shuffle=True, verbose=1)
    prediction = (model.predict(X_test) > 0.5).astype("int32")  # Use threshold 0.5 to classify
    model.summary()
    save_model_results('Artificial Neural Network', y_test, prediction)

def CNN1D(X_train, X_test, y_train, y_test):
    print('1D - Convolutional Neural Network:')
    X_train = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(Conv1D(100, kernel_size=7, input_shape=(1034, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs_number, validation_split=0, shuffle=False, verbose=1)
    prediction = (model.predict(X_test) > 0.5).astype("int32")  # Use threshold 0.5 to classify
    model.summary()
    save_model_results('1D CNN', y_test, prediction)

def CNN2D(X_train, X_test, y_train, y_test):
    print('2D - Convolutional Neural Network:')
    n_array_X_train = X_train.to_numpy()
    n_array_X_train_extended = np.hstack((n_array_X_train, np.zeros(
        (n_array_X_train.shape[0], 2))))

    week = []
    for i in range(n_array_X_train_extended.shape[0]):
        a = np.reshape(n_array_X_train_extended[i], (-1, 7, 1))
        week.append(a)
    X_train_reshaped = np.array(week)

    n_array_X_test = X_test.to_numpy()
    n_array_X_test_extended = np.hstack((n_array_X_test, np.zeros((n_array_X_test.shape[0], 2))))

    week2 = []
    for i in range(n_array_X_test_extended.shape[0]):
        b = np.reshape(n_array_X_test_extended[i], (-1, 7, 1))
        week2.append(b)
    X_test_reshaped = np.array(week2)

    input_shape = (1, 148, 7, 1)

    model = Sequential()
    model.add(Conv2D(kernel_size=(7, 3), filters=32, input_shape=input_shape[1:], activation='relu',
                     data_format='channels_last'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train_reshaped, y_train, validation_split=0.1, epochs=epochs_number, shuffle=False, verbose=1)
    prediction = (model.predict(X_test_reshaped) > 0.5).astype("int32")  # Use threshold 0.5 to classify
    model.summary()
    save_model_results('2D CNN', y_test, prediction)

def LogisticRegressionModel(X_train, X_test, y_train, y_test):
    print('Logistic Regression:')
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    save_model_results('Logistic Regression', y_test, prediction)

def RandomForest(X_train, X_test, y_train, y_test):
    print('Random Forest:')
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    save_model_results('Random Forest', y_test, prediction)

def DecisionTree(X_train, X_test, y_train, y_test):
    print('Decision Tree:')
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    save_model_results('Decision Tree', y_test, prediction)

def SVM(X_train, X_test, y_train, y_test):
    print('Support Vector Machine:')
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    save_model_results('SVM', y_test, prediction)

# Main execution
X_train, X_test, y_train, y_test = read_data()

# Model Execution
# ANN(X_train, X_test, y_train, y_test)
# CNN1D(X_train, X_test, y_train, y_test)
# CNN2D(X_train, X_test, y_train, y_test)
# LogisticRegressionModel(X_train, X_test, y_train, y_test)
# RandomForest(X_train, X_test, y_train, y_test)
# DecisionTree(X_train, X_test, y_train, y_test)
SVM(X_train, X_test, y_train, y_test)
