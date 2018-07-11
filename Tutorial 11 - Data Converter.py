import pickle

data = pickle.load(open("Tutorial 11.pickle", "rb"))
data.to_csv("Tutorial 11.csv")