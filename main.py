import numpy as np
import pandas as pd
import scipy
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

def main():
    #books = [i.strip().split(",") for i in open('/archive/Books.csv', 'r').readlines()]
    #users_list = [i.strip().split(",") for i in open('/archive/nickbecker/Downloads/ml-1m/users.dat', 'r').readlines()]
    #movies_list = [i.strip().split(",") for i in open('/archive/nickbecker/Downloads/ml-1m/movies.dat', 'r').readlines()]

    ratings_df = pd.read_csv("archive/ratings_new.csv")
    print(ratings_df.head(5))
    print("dataframe size:", ratings_df.size)


    reader = Reader(rating_scale=(1, 10))

    dataset = Dataset.load_from_df(ratings_df[["User-ID", "ISBN", "Book-Rating"]], reader)

    #param_grid = {"n_factors" : 100, "n_epochs": [5, 10], "lr_all": [0.001, 0.002, 0.005], "reg_all": [0.4, 0.6]}

    #cross_validate(NormalPredictor(), dataset, cv=2)

    algorithm = SVD(n_factors=50, n_epochs=15, verbose=True)

    cross_validate(algorithm, dataset, measures=["RMSE", "MAE"], cv=5, verbose=True)



if __name__ == "__main__":
    main()