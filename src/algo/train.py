from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import load_model
import time
from src.algo.build_generators import build_generators
from tensorflow.python.keras import callbacks
import conf as config

from inspect import getmembers, isfunction
from src.algo import custom_models


def train(workers, data_aug=True):
    train_gen, val_gen, n_batches_train, n_batches_eval = build_generators(data_aug)

    functions = getmembers(custom_models, isfunction)
    function_name = config.cfg.TRAIN_PARAM.FUNCTION
    my_function = None
    for fct in functions:
        if fct[0] == function_name:
            my_function = fct[1]
            break
    if my_function is None: return

    print('training of ' + my_function.__name__)

    model_name = config.cfg.TRAIN_PATH.MODEL_NAME + '/' + function_name +'_RMSProp'
    tensorboard = model_name + '.logs'
    model_name = model_name + '.keras.model'

    # # build the model
    if config.cfg.TRAIN_PATH.RETRAIN is None:
        my_model = my_function(num_output=config.cfg.TRAIN_PARAM.NUM_PARAMETERS, training_size=config.cfg.
                               TRAIN_PARAM.TRAINING_SIZE)
    else:
        my_model = load_model(config.cfg.TRAIN_PATH.PATH_MODEL)

    my_model.summary()
    # callbacks
    cbs = [
        callbacks.ModelCheckpoint(filepath=model_name, monitor='mean_squared_error', verbose=1,
                                  save_best_only=True, save_weights_only=False, mode='min', period=1),

        callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0, patience=10, verbose=1, mode='min'),

        callbacks.TensorBoard(log_dir=tensorboard, histogram_freq=0,
                              batch_size=config.cfg.TRAIN_PARAM.BATCH_SIZE, write_graph=True, write_grads=False,
                              write_images=False),

        callbacks.ReduceLROnPlateau(monitor='mean_squared_error', factor=0.5, patience=10, verbose=1, mode='min',
                                    cooldown=0, min_lr=9e-7)
    ]

    # fit
    if config.cfg.TRAIN_PATH.RETRAIN is None:
        print("model_compile")
        my_model.compile(loss=mean_squared_error, optimizer=RMSprop(lr=1e-4), metrics=[mean_squared_error])
        print("model_generator")
        my_model.fit(
            x=train_gen,
            validation_data=val_gen,
            steps_per_epoch=n_batches_train,
            validation_steps=n_batches_eval,
            epochs=config.cfg.TRAIN_PARAM.NB_EPOCH,
            max_queue_size=40,
            workers=workers,
            use_multiprocessing=config.cfg.TRAIN_PARAM.USE_MULTIPROCESSING,
            callbacks=cbs,
            shuffle=True,
            verbose=1
        )
    else:
        print("model_generator")
        my_model.fit(
            x=train_gen,
            validation_data=val_gen,
            steps_per_epoch=n_batches_train,
            validation_steps=n_batches_eval,
            epochs=config.cfg.TRAIN_PARAM.NB_EPOCH,
            max_queue_size=40,
            workers=workers,
            callbacks=cbs,
            shuffle=True,
            verbose=1
        )


if __name__ == '__main__':
    start_time = time.time()
    train(config.cfg.TRAIN_PARAM.WORKERS)
    # Affichage du temps d execution
    print("Temps d execution : %s secondes ---" % (time.time() - start_time))
