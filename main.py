from solution import Solution
import seaborn as sns
import pandas as pd
from independent_Variables import IndependentVariables


def main ():
    dataset = sns.load_dataset ('tips')

    array = dataset.iloc [:, 0:6].values
    X = IndependentVariables (array)
    Y = dataset.iloc [:, 6].values

    # numberOfHiddenLayers = 
    # numberOfNeurons = 
    # numberOfInputs = 14
    solution = Solution (X, Y)
    solution.classify ()
    
if __name__ == "__main__":
    main ()