from imports import *
images_folder = 'Datasets/celeba/img/'

def data_preprocessing_A1(dataset):
    """
    Our custom preprocessing class, for preparing data for Task A1:
    it calls encapsulated main(dataset)

    """
    #convert raw rgb image to normalized in [0,1] one-hot vectors
    def load_reshape_img(img,grayscale):
        x = img_to_array(img)/255.
        if grayscale==True:
            x = rgb2gray(x)
        x = x.reshape((1,)+x.shape)
        x = x.flatten()
        return x

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    #convert the dataset with features the image names to dataset with features theflattened one hot encoding vectors of pixels
    def create_flattened_dataset(dataset,size,grayscale):
        dataset_copy=dataset.copy()
        for i in dataset:
            img_name=images_folder + i
            img = load_img(img_name, target_size=size)
            img_vector=load_reshape_img(img,grayscale)
            dataset_copy.loc[int(img_name.split('/')[-1].split('.')[0])] = img_vector
        #convert Series of numpy arrays to 2D np
        dataset_np=np.stack(dataset_copy)
        #convert 2D np to pd
        dataset_pd = pd.DataFrame(dataset_np)
        return(dataset_pd)

    def train_dev_test_split(images, labels):
        #train->0.7, dev->0.15, test->0.15
        frac_train=0.7
        frac_test=0.15
        #random state guarantees that every time the function is called, the same split happens in data
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=frac_test, random_state=78)
        frac_dev=0.15/(frac_train+frac_test)
        X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=frac_dev, random_state=78)
        return X_train, X_test, X_dev, y_train, y_test, y_dev

    # bring dataset to (218, 178, 3) shape
    def extract_RGB_features(images, labels):
        X_train_img, X_test_img, X_dev_img, y_train_img, y_test_img, y_dev_img=train_dev_test_split(images, labels)
        # replace label -1 to 0 for resiliency of CNN networks
        y_train_img=y_train_img.replace(-1,0)
        y_dev_img=y_dev_img.replace(-1,0)
        y_test_img=y_test_img.replace(-1,0)
        #reshape to RGB size to feed in the NN
        X_train_img = X_train_img.to_numpy().reshape((X_train_img.shape[0], 218,178, 3))
        X_dev_img = X_dev_img.to_numpy().reshape((X_dev_img.shape[0], 218,178, 3))
        X_test_img = X_test_img.to_numpy().reshape((X_test_img.shape[0], 218,178, 3))
        return X_train_img, X_dev_img, X_test_img, y_train_img, y_dev_img, y_test_img

    def feature_extraction_from_CNN(X_train_img, X_test_img, X_dev_img):
        #load the best model we trained
        print('Load pretrained CNN model for feature extraction...')
        model = load_model('A1/best.augm_model.h5')
        model.load_weights('A1/weights.best.augm_model.hdf5')
        #take final dense layer to project our raw features into this layer's dimensions
        layer_outputs = model.layers[314].output
        activation_model = Model(inputs=model.input, outputs=layer_outputs)
        print('Extract new features from pretrained CNN model...')
        X_train_new = activation_model.predict(X_train_img)
        print("Data Shape Before CNN Feature Extraction:",X_train_img[0].shape)
        print("Data Shape After CNN Feature Extraction:",X_train_new[0].shape)
        X_dev_new = activation_model.predict(X_dev_img)
        X_test_new = activation_model.predict(X_test_img)
        return(X_train_new, X_dev_new, X_test_new)

    def feature_extraction_PCA(X_train_new, X_dev_new, X_test_new):
        print("\n PCA feature Extraction...\n")
        #transform data through PCA technique
        n = 125
        pca = PCA(n_components=n)
        X_train_new_PCA = pca.fit_transform(X_train_new)
        print("Data Shape After PCA Feature Extraction:",X_train_new_PCA[0].shape)
        X_dev_new_PCA = pca.transform(X_dev_new)
        X_test_new_PCA = pca.transform(X_test_new)
        return(X_train_new_PCA, X_dev_new_PCA, X_test_new_PCA)

    #check for NaN values in one-hot encodings ,e.g. missing pixels
    def check_NaN_values(features,_labels):
        _labels=np.array(_labels)
        NaNvaluesData=np.isnan(_labels).sum().sum()
        if NaNvaluesData>0 :
                where_is_NaN = np.isnan(_labels)
                print("NaN value in labels index: ",where_is_NaN)
        else:
            print("No NaN values in Labels.")

        features=np.array(features)
        NaNvaluesData=np.isnan(features).sum().sum()
        if NaNvaluesData>0 :
            where_is_NaN = np.isnan(features)
            print("Number of NaN values in image features: ",where_is_NaN)
        else:
            print("No NaN values in image features.")
        return


    def main(dataset):
        print("\nPreprocess your data...\n")
        images = dataset.iloc[:,0]
        labels = dataset.iloc[:,1]
        # get actual size of data
        size=(218,178)
        grayscale=False
        reshaped_images=create_flattened_dataset(images,size,grayscale)
        check_NaN_values(reshaped_images,labels)
        X_train_img, X_dev_img, X_test_img, y_train_img, y_dev_img, y_test_img = extract_RGB_features(reshaped_images, labels)
        # feature extraction from pretrained CNN
        X_train_new, X_dev_new, X_test_new = feature_extraction_from_CNN(X_train_img, X_test_img, X_dev_img)
        X_train_new_PCA, X_dev_new_PCA, X_test_new_PCA = feature_extraction_PCA(X_train_new, X_dev_new, X_test_new)
        return X_train_new_PCA, X_dev_new_PCA, X_test_new_PCA, y_train_img, y_dev_img, y_test_img



    X_train, X_dev, X_test, y_train, y_dev, y_test = main(dataset)
    return(X_train, X_dev, X_test, y_train, y_dev, y_test)


def Print_Classification_Results(y_dev,y_pred):
    """
    Our custom class, for printing Classificaation Results for Task A1:

    """
    print(classification_report(y_dev, y_pred))
    acc_score = accuracy_score(y_dev, y_pred)
    print("\nAccuracy Score: ","{:.3%}".format(acc_score))
    conf_matrix = confusion_matrix(y_dev, y_pred)

    labels=['F','M']
    sns.heatmap(conf_matrix, annot=True, fmt='g', xticklabels=labels, yticklabels=labels,cmap= "BuPu")
    plt.show()
    return acc_score
