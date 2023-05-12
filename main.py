import numpy as np
import pandas as pd
import random, re
from surprise import accuracy, Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
from collections import defaultdict


#Loads the ratings dataset and splits it into training and testing subsets
def build_datasets(ratings_file_path):
    ratings_df = pd.read_csv(ratings_file_path)
    print(ratings_df.head(5))
    print("dataframe size:", ratings_df.size, "\n")

    reader = Reader(rating_scale=(1, 10))
    dataset = Dataset.load_from_df(ratings_df[["User-ID", "ISBN", "Book-Rating"]], reader)
    ratings = dataset.raw_ratings

    random.shuffle(ratings)

    threshold = int(0.8 * len(ratings))
    train_ratings = ratings[:threshold]
    test_ratings = ratings[threshold:]

    return dataset, train_ratings, test_ratings


#Builds the users dictionary
def build_users(users_file_path):
    users = {}
    with open(users_file_path, "r") as file:
        file.readline()
        lines = file.readlines()
        for line in lines:
            new_line = line.rstrip("\n")
            tokens = new_line.split(",")
            loc_match = re.search(r"\"[^\"]+\"", line)
            loc = loc_match[0].strip("\"")
            age = tokens[-1]
            if age == "":
                age = None
            else:
                age = float(age)
            users[int(tokens[0])] = {
                "location" : loc,
                "age" : age,
            }
    return users


#Builds the books dataset
def build_books(books_file_path):
    books = {}
    with open(books_file_path, "r") as file:
        file.readline()
        lines = file.readlines()
        for line in lines:
            new_line = line.rstrip("\n")
            tokens = new_line.split(",")
            isbn = tokens[0]
            info = new_line.replace(isbn + ",", "")
            urls_match = re.search(r",http[^,]+,http[^,]+,http[^,]+,?$", new_line)
            urls = urls_match[0]
            info = info.replace(urls, "")
            books[isbn] = info
    return books


#Build the user ratings dictionary
def build_user_ratings(ratings_file_path):
    user_ratings = {}
    with open(ratings_file_path, "r") as file:
        file.readline()
        lines = file.readlines()
        for line in lines:
            new_line = line.rstrip("\n")
            tokens = new_line.split(",")
            user_id = int(tokens[0])
            isbn = tokens[1]
            rating = tokens[2]
            user_ratings[user_id] = { isbn : int(rating) }
    return user_ratings


#Used to determine the best hyperparameter values based on the training dataset
def tune_hyperparams(dataset):
    param_grid = {"n_factors" : [50, 75, 100], "n_epochs": [10, 20, 30], "lr_all": [0.001, 0.002, 0.005]}
    gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=5, joblib_verbose=5)
    
    gs.fit(dataset)

    print("RMSE")
    print(gs.best_score["rmse"])
    print(gs.best_params["rmse"])
    print()

    print("MAE")
    print(gs.best_score["mae"])
    print(gs.best_params["mae"])
    print()


#Returns the top n predicted books
def top_n_preds(predictions, n=5):

    top_n = defaultdict(list)
    for user_id, isbn, true_r, est, _ in predictions:
        top_n[user_id].append((isbn, est))

    for user_id, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[user_id] = user_ratings[:n]

    return top_n


#Prints out examples from the testing dataset, num_test_samples is the number
#of test samples to show
def test_examples(top_n_preds, users, books, user_ratings, num_test_samples):
    print("Test Examples")
    count = 0
    for user_id, preds in top_n_preds.items():
        print("user id:", user_id)
        user = users[user_id]
        print(f"location - {user['location']}; age - {user['age']}")
        print("\nranked:")
        for isbn, rating in user_ratings[user_id].items():
            print("isbn -", isbn)
            if isbn in books:
                print(books[isbn])
            print("rating -", rating)
        print("\npredicted:")
        for pred in preds:
            isbn = pred[0]
            print("isbn -", isbn)
            if isbn in books:
                print(books[isbn])
            print("predicted rating -", pred[1])
        if count == (num_test_samples - 1):
            break
        print("-------------------------------------------")
        count += 1


#Does the evaluation for the training and testing datasets, also calls test_examples()
def evaluate(svd, dataset, train, test_ratings, users, books, user_ratings, num_test_samples):

    train_dataset = train.build_testset()

    test_dataset = dataset.construct_testset(test_ratings)

    predictions = svd.test(train_dataset)
    print("Train Dataset")
    accuracy.rmse(predictions)
    accuracy.mae(predictions)
    print()

    predictions = svd.test(test_dataset)
    print("Test Dataset")
    accuracy.rmse(predictions)
    accuracy.mae(predictions)
    print()

    top_n = top_n_preds(predictions, n=3)

    test_examples(top_n, users, books, user_ratings, num_test_samples)


def main(ratings_file_path, users_file_path, books_file_path, num_test_samples):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    dataset, train_ratings, test_ratings = build_datasets(ratings_file_path)
    dataset.raw_ratings = train_ratings
    users = build_users(users_file_path)
    books = build_books(books_file_path)
    user_ratings = build_user_ratings(ratings_file_path)

    #tune_hyperparams(dataset)

    train = dataset.build_full_trainset()

    svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005)
    svd.fit(train)

    evaluate(svd, dataset, train, test_ratings, users, books, user_ratings, num_test_samples)


if __name__ == "__main__":
    main("archive/ratings_new.csv", "archive/Users.csv", "archive/Books.csv", 4)