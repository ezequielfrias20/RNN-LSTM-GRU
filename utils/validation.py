from sklearn.model_selection import KFold
import numpy as np

def validacion_cruzada(X,y,model):
    kf = KFold(n_splits=5)
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
        val_loss = model.evaluate(X_val, y_val, verbose=0)
        print(f"Fold Loss: {val_loss[0]:.4f}")