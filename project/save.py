import pickle


def save_history(history, filename):

    with open(filename, 'wb') as file:

        pickle.dump(history.history, file)
