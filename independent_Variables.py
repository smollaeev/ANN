from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler


class IndependentVariables:
    def __init__ (self, array):
        self.array = array

    def determine_DummyVariables (self):
        onehotencoder = OneHotEncoder (categorical_features = [4])
        self.array = onehotencoder.fit_transform (self.array).toarray ()
        self.array = self.array [:, 1:]

    def encode_CategoricalVariables (self):
        labelencoder_array_1 = LabelEncoder ()
        self.array [:, 2] = labelencoder_array_1.fit_transform (self.array[:, 2])
        labelencoder_array_2 = LabelEncoder ()
        self.array [:, 3] = labelencoder_array_2.fit_transform (self.array[:, 3])
        labelencoder_array_3 = LabelEncoder ()
        self.array [:, 4] = labelencoder_array_3.fit_transform (self.array[:, 4])
        labelencoder_array_4 = LabelEncoder ()
        self.array [:, 5] = labelencoder_array_4.fit_transform (self.array[:, 5])

        self.determine_DummyVariables ()

    def scale_Features (self):
        sc = StandardScaler ()
        self.array = sc.fit_transform (self.array)
