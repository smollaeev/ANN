from solution import Solution
import seaborn as sns
import pandas as pd
from independent_Variables import IndependentVariables


def main ():
    dataset = sns.load_dataset ('tips')

    array = dataset.iloc [:, 0:6].values
    X = IndependentVariables (array)
    Y = dataset.iloc [:, 6].values
    
    solution = Solution (X, Y)
    solution.preprocess_Data ()
    solution.ANN ()
    print (solution.accuracy)

if __name__ == "__main__":
    main ()