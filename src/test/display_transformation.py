import src.utils.plot_points_on_faces as plot_points
from imgaug import augmenters as iaa
from src.utils.utils import normalize_image, normalize_label, encode_5FP, Fliplr_5fp, RedefineBoxes, \
    load_txt_5FP_and_box
from src.algo.batch_generator_semi_supervised import BatchGeneratorSemiSupervised, size_of_train

batch_size = 64
train_txt_file = r'../../data/train_%d_5FP.txt' % size_of_train
val_txt_file = r'../../data/valid_5000_5FP.txt'
image_dir = r'../../data/'
classes = ('face')

train_data_list = load_txt_5FP_and_box(train_txt_file, image_dir)

img_aug_conf = iaa.Sequential([
    iaa.Affine(rotate=(-35, 35)),
    iaa.PerspectiveTransform(scale=(0, 0.10)),
    iaa.CropAndPad(percent=(-0.0, -0.15)),
    iaa.Sometimes(0.5, Fliplr_5fp()),
    iaa.Add((-20, +20), per_channel=True),
    iaa.Sometimes(0.5,
                  iaa.GaussianBlur(sigma=(0, 0.1))
                  )
    ,
    iaa.Sometimes(0.5, iaa.Grayscale(1.0)),
    iaa.Sharpen((0.0, 0.5)),
    RedefineBoxes()
])

img_aug_conf_unsup = iaa.Sequential([
    iaa.Add((-20, +20), per_channel=True),
    iaa.Sometimes(0.5,
                  iaa.GaussianBlur(sigma=(0, 0.1))
                  )
    ,
    iaa.Sometimes(0.5, iaa.Grayscale(1.0)),
    iaa.Sharpen((0.0, 0.5)),
    iaa.Sometimes(0.3, iaa.Cutout(nb_iterations=1)),
    RedefineBoxes()
],
    random_order=False
)

img_aug_conf_valid = iaa.Sequential([
    RedefineBoxes()
],
    random_order=False
)

train_gen = BatchGeneratorSemiSupervised(batch_size=batch_size,
                                         data_list=train_data_list,
                                         classes=classes,
                                         shuffle=True,
                                         image_normalization_fn=normalize_image,
                                         label_normalization_fn=normalize_label,
                                         list_transfo=['stack'],
                                         img_aug_conf=img_aug_conf,
                                         img_aug_conf_unsupervised=img_aug_conf_unsup,
                                         encoding_fn=encode_5FP
                                         )

plot_points.plot_from_generator(train_gen, 64, unsupervised=True)
