from imports import *
from A1.gender_classification import *
from A2.smile_detection import *
from B1.faceShape_recognition import *
from B2.eyeColor_recognition import *

# "D:/Xenia/UCL MSc/Term 1/Applied ML Systems I - ELEC0134/Assignement/AMLS_20-21_SN20159465/A1/model_A1.pkl"
pkl_filename_A1 = "A1/model_A1.pkl"
model_filename_A2 = "A2/best.modelB3.h5"
weights_filename_A2 = "A2/weights.best.modelB3.hdf5"
pkl_filename_B2 = "B2/model_B2.pkl"

# run: python main.py A1 for task A1
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("\nYou chose to run all tasks...Let's start!\n")
        task = 'all'
    else:
        task = str(sys.argv[1])
        if task == 'A1' or task == 'a1' or task == 'B1' or task == 'b1' or task == 'A2' or task == 'a2' or task == 'B2' or task == 'b2':
            print("\nYou chose to run Task {}...Let's start!\n".format(str(sys.argv[1])))
        else:
            print("\nThe task name you entered doesn't exist...Try again!")

    # ======================================================================================================================
    # Task A1
    if task  == 'A1' or task  == 'a1' or task == 'all':
        print('Task A1 running...\n')
        # Load Dataset for Task A1
        dataset = pd.read_csv('Datasets/celeba/labels.csv', sep = '\t', usecols = [1,2])
        print("Celeba Gender Dataset Size: ",dataset.shape)
        # Data preprocessing
        X_train_A1, X_dev_A1, X_test_A1, y_train_A1, y_dev_A1, y_test_A1 = data_preprocessing_A1(dataset)
        # Load model object A1 which was fine-tuned based on validation set
        print('\nLoad best model A1...\n')
        with open(pkl_filename_A1, 'rb') as file:
            model_A1 = pickle.load(file)
        #Predict target values and Calculate the accuracy score
        # On Train Set:
        print('\nPredict Train Set targets...\n')
        y_pred = model_A1.predict(X_train_A1)
        fig = plt.figure(0)
        fig.canvas.set_window_title('Confusion Matrix for Train Data')
        acc_A1_train = Print_Classification_Results(y_train_A1,y_pred)
        # On Dev Set:
        print('\nPredict Dev Set targets...\n')
        y_pred = model_A1.predict(X_dev_A1)
        fig = plt.figure(1)
        fig.canvas.set_window_title('Confusion Matrix for Dev Data')
        acc_A1_dev = Print_Classification_Results(y_dev_A1,y_pred)
        # On Test Set:
        print('\nPredict Test Set targets...\n')
        y_pred = model_A1.predict(X_test_A1)    # Test model based on the test set.
        fig = plt.figure(2)
        fig.canvas.set_window_title('Confusion Matrix for Test Data')
        acc_A1_test = Print_Classification_Results(y_test_A1,y_pred)
        print("\n")
    # ======================================================================================================================


    # ======================================================================================================================
    # # Task A2
    if task  == 'A2' or task  == 'a2' or task == 'all':
        print('Task A2 running...\n')
        # Load Dataset for Task A2
        dataset = pd.read_csv('Datasets/celeba/labels.csv', sep = '\t', usecols = [1,3])
        print("Celeba Smile Dataset Size: ",dataset.shape)
        # Data preprocessing
        X_train_A2, X_dev_A2, X_test_A2, y_train_A2, y_dev_A2, y_test_A2 = data_preprocessing_A2(dataset)
        # Load model object A1 which was fine-tuned based on validation set
        print('\nLoad best model A2...\n')
        model_A2 = load_model(model_filename_A2)
        model_A2.load_weights(weights_filename_A2)
        #Predict target values and Calculate the accuracy score
        # On Train Set:
        print('\nPredict Train Set targets...\n')
        y_pred = model_A2.predict_classes(X_train_A2)
        fig = plt.figure(0)
        fig.canvas.set_window_title('Confusion Matrix for Train Data')
        acc_A2_train = Print_Classification_Results(y_train_A2,y_pred)
        # On Dev Set:
        print('\nPredict Dev Set targets...\n')
        y_pred = model_A2.predict_classes(X_dev_A2)
        fig = plt.figure(1)
        fig.canvas.set_window_title('Confusion Matrix for Dev Data')
        acc_A2_dev = Print_Classification_Results(y_dev_A2,y_pred)
        # On Test Set:
        print('\nPredict Test Set targets...\n')
        y_pred = model_A2.predict_classes(X_test_A2)    # Test model based on the test set.
        fig = plt.figure(2)
        fig.canvas.set_window_title('Confusion Matrix for Test Data')
        acc_A2_test = Print_Classification_Results(y_test_A2,y_pred)
        print("\n")
    # ======================================================================================================================



    # ======================================================================================================================
    # # Task B1
    if task  == 'B1' or task  == 'b1' or task == 'all':
        print('Task B1 running...\n')
        # Load Dataset for Task B1
        dataset = pd.read_csv('Datasets/cartoon_set/labels.csv', sep = '\t', usecols = [2,3])
        print("Cartoon Face Shape Dataset Size: ",dataset.shape)
        # Data preprocessing
        X_train_B1, X_dev_B1, X_test_B1, y_train_B1, y_dev_B1, y_test_B1 = data_preprocessing_B1(dataset)
        # Load model object A1 which was fine-tuned based on validation set
        print('\nTrain model B1...')
        model_B1 = SVC(kernel='linear',gamma='auto', C= 1.0)
        model_B1.fit(X_train_B1, y_train_B1)
        #Predict target values and Calculate the accuracy score
        # On Train Set:
        print('\nPredict Train Set targets...')
        y_pred = model_B1.predict(X_train_B1)
        fig = plt.figure(0)
        fig.canvas.set_window_title('Confusion Matrix for Train Data')
        acc_B1_train = Print_Classification_Results(y_train_B1,y_pred)
        # On Dev Set:
        print('\nPredict Dev Set targets...')
        y_pred = model_B1.predict(X_dev_B1)
        fig = plt.figure(1)
        fig.canvas.set_window_title('Confusion Matrix for Dev Data')
        acc_B1_dev = Print_Classification_Results(y_dev_B1,y_pred)
        # On Test Set:
        print('\nPredict Test Set targets...\n')
        y_pred = model_B1.predict(X_test_B1)    # Test model based on the test set.
        fig = plt.figure(2)
        fig.canvas.set_window_title('Confusion Matrix for Test Data')
        acc_B1_test = Print_Classification_Results(y_test_B1,y_pred)
        print("\n")
    # ======================================================================================================================



    # # ======================================================================================================================
    # # Task B2
    if task  == 'B2' or task  == 'b2' or task == 'all':
        print('Task B2 running...\n')
        # Load Dataset for Task B2
        dataset = pd.read_csv('Datasets/cartoon_set/labels.csv', sep = '\t', usecols = [1,3])
        print("Cartoon Eye Color Dataset Size: ",dataset.shape)
        # Data preprocessing
        X_train_B2, X_dev_B2, X_test_B2, y_train_B2, y_dev_B2, y_test_B2 = data_preprocessing_B2(dataset)
        # Load model object A1 which was fine-tuned based on validation set
        print('\nLoad best model B2...\n')
        with open(pkl_filename_B2, 'rb') as file:
            model_B2 = pickle.load(file)
        #Predict target values and Calculate the accuracy score
        # On Train Set:
        print('\nPredict Train Set targets...\n')
        y_pred = model_B2.predict(X_train_B2)
        fig = plt.figure(0)
        fig.canvas.set_window_title('Confusion Matrix for Train Data')
        acc_B2_train = Print_Classification_Results(y_train_B2,y_pred)
        # On Dev Set:
        print('\nPredict Dev Set targets...\n')
        y_pred = model_B2.predict(X_dev_B2)
        fig = plt.figure(1)
        fig.canvas.set_window_title('Confusion Matrix for Dev Data')
        acc_B2_dev = Print_Classification_Results(y_dev_B2,y_pred)
        # On Test Set:
        print('\nPredict Test Set targets...\n')
        y_pred = model_B2.predict(X_test_B2)    # Test model based on the test set.
        fig = plt.figure(2)
        fig.canvas.set_window_title('Confusion Matrix for Test Data')
        acc_B2_test = Print_Classification_Results(y_test_B2,y_pred)
        print("\n")
    # # ======================================================================================================================


    # ## Print out your results with following format:
    if task == 'all':
        print('TA1:{},{}\nTA2:{},{}\nTB1:{},{}\nTB2:{},{}'.format(acc_A1_train, acc_A1_test,
                                                                acc_A2_train, acc_A2_test,
                                                                acc_B1_train, acc_B1_test,
                                                                acc_B2_train, acc_B2_test))
