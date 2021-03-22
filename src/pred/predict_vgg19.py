from imgaug import augmenters as iaa
import src.utils.utils as utils
import src.algo.batch_generator_supervised as sup_batch
import src.utils.config as config
import src.utils.plot_points_on_faces as plot_points
import time
from src.pred.utils_predict import load_my_model, predict_and_compute_losses


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

#"../../models/vgg19/new/train_vgg19_2021_03_22_19_50_42_10000_valid_5000.keras.model"
#"../../models/mobilenet_v2/new/train_supervised_2021_03_21_23_31_25_10000_valid_5000.keras.model"

# Main
if __name__ == '__main__':
    a = time.time()

    my_model = load_my_model()
    prediction, mse, eyes_nose_lips_mse = predict_and_compute_losses(my_model, val_gen)
    print('mse', mse)
    plot_points.plot_from_generator(val_gen, 10, prediction, True)

    print('Temps d execution en secondes :', time.time() - a)
