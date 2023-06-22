import numpy as np

def load_weigths():
    with open("wnet0.npy", 'rb') as f:
        return np.load(f)


def load_test():
    X = []
    with open("testnet0.txt") as file:
        for line in file:
            if line.strip() != "":
                string = line.strip()
                input_with_bias = np.concatenate(
                    ([1], np.array([list(letter) for letter in string], dtype=int).flatten()))
                X.append(input_with_bias)
    return X


def feed_forward(w, X):
    preds = []
    for x in X:
        pred = np.dot(x, w)[0]
        if pred > 0:
            preds.append(1)
        else:
            preds.append(0)

    return preds


def save_preds(preds):
    with open("result.txt", 'w') as f:
        for p in  preds:
            f.write(str(p) + "\n")


if __name__ == "__main__":
    w = load_weigths()
    X = load_test()
    preds = feed_forward(w, X)
    save_preds(preds)