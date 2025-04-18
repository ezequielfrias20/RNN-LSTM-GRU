from tensorflow.keras.callbacks import EarlyStopping


def entrenamiento(model, X_train, y_train, X_val, y_val, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT):

    early_stop = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop],
        # Controla cuánta información se muestra durante el entrenamiento. 1: Muestra barra de progreso y métricas por época (recomendado).
        verbose=1
    )
    return history