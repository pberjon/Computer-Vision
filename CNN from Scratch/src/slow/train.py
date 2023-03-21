from src.slow.layers import *
from src.slow.utils import *
from src.slow.model import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import trange

filename = [
        ["training_images","train-images-idx3-ubyte.gz"],
        ["test_images","t10k-images-idx3-ubyte.gz"],
        ["training_labels","train-labels-idx1-ubyte.gz"],
        ["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def train():
    print("\n----------------EXTRACTION---------------\n")
    X, y, X_test, y_test = load(filename)
    X, X_test = X/float(255), X_test/float(255)
    X -= np.mean(X)
    X_test -= np.mean(X_test)

    print("\n--------------PREPROCESSING--------------\n")
    X = resize_dataset(X)
    print("Resize dataset: OK")
    y = one_hot_encoding(y)
    print("One-Hot-Encoding: OK")
    X_train, y_train, X_val, y_val = train_val_split(X, y)
    print("Train and Validation set split: OK\n")

    model = LeNet5()
    cost = CrossEntropyLoss()
    
    params = model.get_params()

    optimizer = AdamGD(lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, params = model.get_params())    
    train_costs, val_costs = [], []
    
    print("----------------TRAINING-----------------\n")

    NB_EPOCH = 1
    BATCH_SIZE = 100

    print("EPOCHS: {}".format(NB_EPOCH))
    print("BATCH_SIZE: {}".format(BATCH_SIZE))
    print()

    nb_train_examples = len(X_train)
    nb_val_examples = len(X_val)

    best_val_loss = float('inf')


    for epoch in range(NB_EPOCH):

        #-------------------------------------------------------------------------------
        #                                       
        #                               TRAINING PART
        #
        #-------------------------------------------------------------------------------
        
        train_loss = 0
        train_acc = 0 

        pbar = trange(nb_train_examples // BATCH_SIZE)
        train_loader = dataloader(X_train, y_train, BATCH_SIZE)

        for i, (X_batch, y_batch) in zip(pbar, train_loader):
           
            y_pred = model.forward(X_batch)
            loss = cost.get(y_pred, y_batch)
            
            grads = model.backward(y_pred, y_batch)
            params = optimizer.update_params(grads)
            model.set_params(params)

            train_loss += loss * BATCH_SIZE
            train_acc += sum((np.argmax(y_batch, axis=1) == np.argmax(y_pred, axis=1)))

            pbar.set_description("[Train] Epoch {}".format(epoch+1))
        
        train_loss /= nb_train_examples
        train_costs.append(train_loss)
        train_acc /= nb_train_examples

        info_train = "train-loss: {:0.6f} | train-acc: {:0.3f}"
        print(info_train.format(train_loss, train_acc))

        #-------------------------------------------------------------------------------
        #                                       
        #                               VALIDATION PART
        #
        #-------------------------------------------------------------------------------
        val_loss = 0
        val_acc = 0 

        pbar = trange(nb_val_examples // BATCH_SIZE)
        val_loader = dataloader(X_val, y_val, BATCH_SIZE)

        for i, (X_batch, y_batch) in zip(pbar, val_loader):

            y_pred = model.forward(X_batch)
            loss, deltaL = cost.get(y_pred, y_batch)
            
            grads = model.backward(deltaL)
            params = optimizer.update_params(grads)
            model.set_params(params)

            val_loss += loss * BATCH_SIZE
            val_acc += sum((np.argmax(y_batch, axis=1) == np.argmax(y_pred, axis=1)))

            pbar.set_description("[Val] Epoch {}".format(epoch+1))

        val_loss /= nb_val_examples
        val_costs.append(val_loss)
        val_acc /= nb_val_examples

        info_val =  "val-loss: {:0.6f} | val-acc: {:0.3f}"
        print(info_val.format(val_loss, val_acc))

        if best_val_loss > val_loss:
            print("Validation loss decreased from {:0.6f} to {:0.6f}. Model saved".format(best_val_loss, val_loss))
            save_params_to_file(model)
            best_val_loss = val_loss

        print()

    pbar.close()

train()
