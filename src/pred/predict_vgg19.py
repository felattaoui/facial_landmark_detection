from imgaug import augmenters as iaa
import src.utils.utils as utils
import src.algo.batch_generator_supervised as sup_batch
import src.utils.config as config
import src.utils.plot_points_on_faces as plot_points
import time
from src.pred.utils_predict import load_my_model, predict_and_compute_losses
import numpy as np


CLASSES = ('face')

test_data_list = utils.load_txt_5FP_and_box(config.cfg.TRAIN_PATH.VAL_TXT_FILE, config.cfg.TRAIN_PATH.IMAGE_DIR)

# Prediction Generator
print("predict generator")
img_aug_conf_valid = iaa.Sequential([utils.RedefineBoxes()], random_order=False)

print("test generator")
val_gen = sup_batch.BatchGeneratorSupervised(batch_size=config.cfg.TRAIN_PARAM.BATCH_SIZE,
                                             training_size=config.cfg.TRAIN_PARAM.TRAINING_SIZE,
                                             data_list=test_data_list,
                                             classes=CLASSES,
                                             shuffle=False,
                                             image_normalization_fn=utils.normalize_image,
                                             label_normalization_fn=utils.normalize_label,
                                             img_aug_conf=img_aug_conf_valid,
                                             encoding_fn=utils.encode_5FP,
                                             preprocess_pred=True
                                             )


def predict_images_one_by_one(generator, model):
    nb_images_plotted=0
    total_time = 0
    for batch_img, batch_label in generator:
        for img, label in zip(batch_img, batch_label):
            # print(img.shape)
            # img = from_BGR_to_RGB(img)
            #img = denorm_image_mv2(img)
            # img_expanded = np.expand_dims(img, axis=0)
            # print(img_expanded.shape)
            # label = denorm_labels(label)
            a = time.time()
            img_expanded = np.expand_dims(img, axis=0)
            pred = model.predict(img_expanded)
            b = time.time()-a
            total_time +=b
            # pred = np.squeeze(pred)
            # pred = denorm_labels(pred)
            # img = denorm_image_mv2(img)
            #plot_points_on_face(img, label, pred)
            # nb_images_plotted += 1
            # if nb_images_plotted > 1:
            #     break
            # break
    print('total time', total_time)


# Main
if __name__ == '__main__':
    a = time.time()
    #model_path = "../../trained_models/train_vgg19_2021_03_23_23_23_12_10000_valid_5000.keras.model"
    #model_path = "../../models/vgg19/new/train_vgg19_2021_03_29_02_24_47_10000_valid_5000.keras.model"
    model_path = "../../models/vgg19/new/train_vgg19_2021_03_24_23_33_41_10000_valid_5000.keras.model"
    my_model = load_my_model(model_path)
    my_model.evaluate(val_gen)
    # prediction, mse, eyes_nose_lips_mse = predict_and_compute_losses(my_model, val_gen)
    # print('mse', mse)
    # plot_points.plot_from_generator(val_gen, 10, prediction, True)

    print('Temps d execution en secondes :', time.time() - a)
