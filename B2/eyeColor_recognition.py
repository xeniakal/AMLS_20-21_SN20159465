from imports import *
images_folder = 'Datasets/cartoon_set/img/'
offset_height=40
offset_width=30
target_height=30
target_width=50


def data_preprocessing_B2(dataset):
    """
    Our custom preprocessing class, for preparing data for Task B2:
    it calls encapsulated main(dataset)

    """
    #convert raw rgb image to normalized in [0,1] one-hot vectors
    def load_reshape_img(img):
        x = img_to_array(img)/255.
        x = x.reshape((1,)+x.shape)
        x = x.flatten()
        return x

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    #convert the dataset with features the image names to dataset with features theflattened one hot encoding vectors of pixels
    def create_flattened_dataset(dataset,size,crop_bool):
        dataset_copy=dataset.copy()
        for i in dataset:
            img_name=images_folder + i
            img = load_img(img_name, target_size=size)
            #cropped image size = (30,50,3)
            if crop_bool == True:
                x=img_to_array(img)
                x_cropped = tf.image.crop_to_bounding_box(x, offset_height, offset_width, target_height, target_width)
                x_cropped =tf.keras.preprocessing.image.array_to_img(x_cropped, data_format=None, scale=True, dtype=None)
                img_vector=load_reshape_img(x_cropped)
            else:
                img_vector=load_reshape_img(img)
            dataset_copy.loc[int(img_name.split('/')[-1].split('.')[0])] = img_vector
        dataset_np=np.stack(dataset_copy)
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
            print("No NaN values in Image features.")
        return

    def feature_extraction_PCA(X_train, X_dev, X_test):
        print("\n PCA feature Extraction...\n")
        #transform data through PCA technique
        n = 300
        pca = PCA(n_components=n)
        X_train_PCA = pca.fit_transform(X_train)
        print("Cropped Data Shape Before PCA Feature Extraction:",X_train[0].shape)
        print("Cropped Data Shape After PCA Feature Extraction:",X_train_PCA[0].shape)
        X_dev_PCA = pca.transform(X_dev)
        X_test_PCA = pca.transform(X_test)
        return(X_train_PCA, X_dev_PCA, X_test_PCA)


    #function for SVM classifier with parameters of kernel, gamma and C
    def Print_Classification_Results(y_dev,y_pred):
        print(classification_report(y_dev, y_pred))
        acc_score = accuracy_score(y_dev, y_pred)
        print("\nAccuracy Score: ","{:.3%}".format(acc_score))
        conf_matrix = confusion_matrix(y_dev, y_pred)
        labels=['0','1','2','3','4']
        sns.heatmap(conf_matrix, annot=True, fmt='g', xticklabels=labels, yticklabels=labels,cmap= "BuPu")
        plt.show()
        return acc_score


    def main(dataset):
        print("\nPreprocess your data...\n")
        images = dataset.iloc[:,1]
        labels = dataset.iloc[:,0]
        # get reduced size of data after experiments
        size=(110,110)
        crop_bool = True
        reshaped_images=create_flattened_dataset(images,size,crop_bool)
        check_NaN_values(reshaped_images,labels)
        X_train, X_test, X_dev, y_train, y_test, y_dev=train_dev_test_split(reshaped_images, labels)
        X_train_PCA, X_dev_PCA, X_test_PCA = feature_extraction_PCA(X_train, X_dev, X_test)
        return X_train_PCA, X_dev_PCA, X_test_PCA, y_train, y_dev, y_test


    X_train, X_dev, X_test, y_train, y_dev, y_test = main(dataset)
    return(X_train, X_dev, X_test, y_train, y_dev, y_test)
