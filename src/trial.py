from surprise import KNNBasic
from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf


# Load the movielens-100k dataset (download it if needed),
# and split it into 3 folds for cross-validation.
data = Dataset.load_builtin('ml-100k')
print(data)
data.split(n_folds=3)

algo = KNNBasic()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print_perf(perf)
