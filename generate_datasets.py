
#Creates a new file that removes all the implicit ratings (rating = 0) from Ratings.csv
def generate_new_ratings(file_path, new_file_path):
    with open(new_file_path, "w") as write_file:
        with open(file_path, "r") as read_file:
            lines = read_file.readlines()
            count = 0
            for line in lines:
                new_line = line.rstrip("\n")
                tokens = new_line.split(",")
                if count == 0:
                    write_file.write(line)
                else:
                    if int(tokens[-1]) == 0:
                        continue
                    else:
                        write_file.write(line)
                count += 1

def count_num_books_in_ratings(file_path):
    books = {}
    users = {}
    with open(file_path, "r") as read_file:
        lines = read_file.readlines()
        count = 0
        for line in lines:
            new_line = line.rstrip("\n")
            tokens = new_line.split(",")
            if count == 0:
                count += 1
                continue
            else:
                books[tokens[1]] = 1
                users[int(tokens[1])] = 1
                count += 1
            
    print("num books", len(books.keys()))
    print("num users", len(users.keys()))



if __name__ == "__main__":
    #generate_new_ratings("archive/Ratings.csv", "archive/ratings_new.csv")
    count_num_books_in_ratings("archive/Ratings.csv")