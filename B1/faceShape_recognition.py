from imports import *
images_folder = '../Datasets/cartoon_set/img/'


def data_preprocessing_B1(dataset):
    """
    Our custom preprocessing class, for preparing data for Task B1:
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


    def feature_selection(images,variance_lower_limit):
        selector = VarianceThreshold(variance_lower_limit)
        images_reduced = selector.fit_transform(images)
        print("Images original shape before variance filtering: ",images.shape)
        print("Images reduced shape after variance filtering: ",images_reduced.shape)
        return images_reduced

    def main(dataset):
        print("\nPreprocess your data...\n")
        images = dataset.iloc[:,1]
        labels = dataset.iloc[:,0]
        # get reduced size of data after experiments
        size=(110,110)
        grayscale=True
        reshaped_images=create_flattened_dataset(images,size,grayscale)
        check_NaN_values(reshaped_images,labels)
        variance_lower_limit=0
        reshaped_images_reduced=feature_selection(reshaped_images,variance_lower_limit)
        X_train, X_test, X_dev, y_train, y_test, y_dev=train_dev_test_split(reshaped_images_reduced, labels)
        return X_train, X_dev, X_test, y_train, y_dev, y_test


    X_train, X_dev, X_test, y_train, y_dev, y_test = main(dataset)
    return(X_train, X_dev, X_test, y_train, y_dev, y_test)


    def Print_Classification_Results(y_dev,y_pred):
        print(classification_report(y_dev, y_pred))
        acc_score = accuracy_score(y_dev, y_pred)
        print("\nAccuracy Score: ","{:.3%}".format(acc_score))
        conf_matrix = confusion_matrix(y_dev, y_pred)
        labels=['0','1','2','3','4']
        sns.heatmap(conf_matrix, annot=True, fmt='g', xticklabels=labels, yticklabels=labels,cmap= "BuPu")
        plt.show()
        return acc_score
