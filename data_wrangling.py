#################################################################
##  Script Info:
##  Author: Shirish Pandagare
##  Date : 10/18/2019
#################################################################



import pandas as pd
from sklearn.preprocessing import StandardScaler , MinMaxScaler


class Preprocess:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path  = test_path
        self.train = None
        self.target = None
        self.train_scaled = None
        self.test = None
        self.test_scaled = None
        self.columns_to_drop = None
        self.mean_value = None
        self.median_value = None

    def data_type(self, x):
        if x == 'na':
            return None
        else:
            return float( x )


    def load_data(self):
        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)

        self.train = self.train.applymap(lambda x: self.data_type(x))
        self.test = self.test.applymap(lambda x: self.data_type(x))

        self.target = self.train[['target']]
        self.train = self.train.drop(['target'], axis = 1)

        print("Train data loaded")
        print( "Test data loaded" )

        return self.train , self.target, self.test

    def mean_median(self, data):
        self.mean_value = {}
        for col, val in zip( data.columns, data.mean() ):
            self.mean_value[col] = val

        self.median_value = {}
        for col, val in zip( data.columns, data.median() ):
            self.median_value[col] = val

        return self.mean_value, self.median_value


    def imputation(self, method = 'mean'):
        self.columns_to_drop = self.train.isna().sum()[self.train.isna().sum() > self.train.shape[0] / 2].index
        self.train = self.train.drop(self.columns_to_drop, axis =1 )
        self.test = self.test.drop(self.columns_to_drop, axis =1 )

        self.mean_value , self.median_value = self.mean_median(self.train)

        if method == 'mean':
            self.train = self.train.fillna(self.mean_value)
            self.test = self.test.fillna(self.mean_value)

        elif method == "median":
            self.train = self.train.fillna( self.median_value )
            self.test = self.test.fillna( self.median_value )

        print("Shape of Train Dataset = ", self.train.shape)
        print("Shape of Test Dataset = ", self.test.shape)
        print("Train and Test imputed by {} method".format(method))

        return self.train, self.test

    def scaling(self, method = "Std"):

        if method == "Std":
            scaler = StandardScaler()
            scaler.fit(self.train)
            self.train_scaled = pd.DataFrame(scaler.transform(self.train) , columns= self.train.columns)
            self.test_scaled = pd.DataFrame(scaler.transform(self.test), columns= self.test.columns)

        elif method == "min_max":
            min_max_scaler = MinMaxScaler()
            min_max_scaler.fit(self.train)
            self.train_scaled = pd.DataFrame( min_max_scaler.transform(self.train), columns=self.train.columns )
            self.test_scaled = pd.DataFrame( min_max_scaler.transform(self.test), columns=self.test.columns )


        print("Test and Train are scaled using {}".format(method))

        return self.train_scaled , self.test_scaled


def main():
    prep = Preprocess(train_path= "equip_failures_training_set.csv", test_path= "equip_failures_test_set.csv")
    train , target , test = prep.load_data()
    train , test = prep.imputation(method= 'mean')
    train_scl , test_scl = prep.scaling(method='Std')

    return train_scl , target , test_scl



if __name__ == "__main__":

    main()









