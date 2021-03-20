import cv2
import numpy as np
import os
import imgaug as ia
import imgaug.augmenters as iaa
from tensorflow.keras.utils import Sequence

size_of_train = 100
BATCH_SIZE = 64
TRANSFO_TO_APPLY = ['translation']
NUM_TRANSFO = len(TRANSFO_TO_APPLY)
size_of_train = 100
NB_IMG_LABEL = int(size_of_train / 10)
BATCH_SIZE_LABELIZE = int(BATCH_SIZE / 4)
BATCH_SIZE_UNLABELIZE = int((BATCH_SIZE - BATCH_SIZE_LABELIZE) / (NUM_TRANSFO + 1))


class BatchGeneratorSemiSupervised(Sequence):
    def __init__(self, batch_size, training_size, data_list, classes, shuffle=True, image_normalization_fn=None,
                 label_normalization_fn=None, list_transfo=None, img_aug_conf=None, img_aug_conf2=None,
                 img_aug_conf_unsupervised=None,
                 encoding_fn=None, zipfile=None):
        if list_transfo is None:
            list_transfo = ['translation']
        self.list_transfo = list_transfo
        self.batch_size = batch_size
        self.training_size = training_size
        self.batch_size_labelize = BATCH_SIZE_LABELIZE
        self.batch_size_unlabelize = BATCH_SIZE_UNLABELIZE
        self.data_list = data_list
        self.labelised_list = data_list[:NB_IMG_LABEL]
        self.unlabelised_list = data_list[NB_IMG_LABEL:]
        self.classes = classes
        self.image_normalization_fn = image_normalization_fn
        self.label_normalization_fn = label_normalization_fn
        self.ia_conf = img_aug_conf
        self.ia_conf_unsupervised = img_aug_conf_unsupervised
        self.ia_conf2 = img_aug_conf2
        self.encoding_fn = encoding_fn
        self.shuffle = shuffle
        self.seeded = False

    def __len__(self):
        return int(np.ceil(self.size() / self.batch_size))

    def num_classes(self):
        return len(self.labels)

    def size(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx_l = idx % (len(self.labelised_list) // self.batch_size_labelize - 1)
        idx_ul = idx % (len(self.unlabelised_list) // self.batch_size_unlabelize - 1)
        if not self.seeded:
            seed = int.from_bytes(os.urandom(4), byteorder='little')
            ia.seed(seed)
            self.ia_conf.reseed(seed)
            self.seeded = True
        sub_dl_l = self.labelised_list[idx_l * self.batch_size_labelize:(idx_l + 1) * self.batch_size_labelize:]
        sub_dl_ul = self.unlabelised_list[
                    idx_ul * self.batch_size_unlabelize:(idx_ul + 1) * self.batch_size_unlabelize:]
        images_l = self.load_images(self, sub_dl_l)
        labels = self.load_labels(sub_dl_l)
        images_ul = self.load_images(self, sub_dl_ul)
        labels_ul = self.load_labels(sub_dl_ul)
        # Creation image transformé et ajout de plusieurs lignes de label pour permettre le cropping
        images_transform, labels_transform, labels_ul = self.generate_unlabelised_transformation(images_ul, labels_ul)
        # Data Augmentation et cropping des images
        # Create a empty array to fit the size of the images that will be crop
        images_ul = np.zeros(
            (images_transform.shape[0], self.training_size, self.training_size, images_transform.shape[3]))
        if self.ia_conf is not None:
            for i in range(len(images_l)):
                images_l[i], labels[i] = self.augment_image_and_labels(images_l[i], labels[i])
            for i in range(len(images_ul)):
                images_ul[i], _ = self.augment_image_and_labels(images_transform[i].astype("uint8"), labels_ul[i],
                                                                unsupervised=True)
        # Normalisation des images
        if self.image_normalization_fn is not None:
            images_l = [self.image_normalization_fn(img) for img in images_l]
            images_ul = [self.image_normalization_fn(img) for img in images_ul]
        # Encoding des images
        if self.encoding_fn is not None:
            for i in range(len(images_l)):
                images_l[i], labels[i] = self.encoding_fn(images_l[i], labels[i])
            images_ul = np.array(images_ul)
            images_l = np.array(images_l)
            labels = np.array(labels)
        # Normalisation des labels
        if self.label_normalization_fn is not None:
            labels = self.label_normalization_fn(labels, self.training_size)
            # labels_ul = self.label_normalization_fn(labels_transform, TRAINING_SIZE)
        # Concatenation des images labelisé et non labelisé
        images = np.vstack((images_l, images_ul))
        labels = np.vstack((labels, labels_transform))
        return images, labels

    @staticmethod
    def load_images(self, data_list):
        images = [cv2.imread(d['filename']) for d in data_list]
        return images

    def load_labels(self, data_list):
        labels = []
        for d in data_list:
            labels.append([{'class': self.classes.index(o['class']),
                            '5FP': o['5FP'], 'face_box': o['face_box']} for o in d['objects'] if
                           o['class'] in self.classes])
        return labels

    def generate_unlabelised_transformation(self, images_ul, labels_ul):
        """
        Generate all transformation in list_transfo for all image present in images_ul.
        Add a additional row per transformation in labels_ul to keep the same dimension and enable cropping
        :param images_ul:
        :param labels_ul:
        :return:
        """
        images_ul = np.array(images_ul)
        images_transform = np.zeros((images_ul.shape[0] * (len(self.list_transfo) + 1), images_ul.shape[1],
                                     images_ul.shape[2], images_ul.shape[3]))
        # label for cropping
        labels_ul_resize = []
        # -1 for discrimination in the loss function
        labels_transform = -2 * np.ones((images_ul.shape[0] * (len(self.list_transfo) + 1), 10))
        for image_idx, i in enumerate(range(0, images_transform.shape[0], len(self.list_transfo) + 1)):
            images_transform[i] = images_ul[image_idx]
            labels_ul_resize.append(labels_ul[image_idx])
            for j in range(len(self.list_transfo)):
                image_transform, label_transform = self.transform_unlabelised(images_ul[image_idx],
                                                                              labels_ul[image_idx],
                                                                              type=self.list_transfo[j])
                images_transform[i + j + 1] = image_transform
                labels_transform[i + j + 1] = label_transform
                labels_ul_resize.append(labels_ul[image_idx])
        return images_transform, labels_transform, labels_ul_resize

    def transform_unlabelised(self, image, label, type='translation'):
        """
        Generate a random transformation from the type given
        :param image:
        :param type:
        :return: the transforme image and the label containing [-2, idx_of_transfo, value_of_transfo, ...] of size 10
        idx_of_transfo is 0 for translation and 1 for rotation
        """

        def generate_translation():
            translate_value_x = np.random.randint(-25, 26, 1)[0]  # random entre -25/ 25
            translate_value_y = np.random.randint(-25, 26, 1)[0]  # random entre -25/ 25
            aug = iaa.Affine(
                translate_px={"x": -translate_value_x, "y": -translate_value_y})
            return translate_value_x, translate_value_y, aug

        def generate_rotation():
            rotate_value = np.random.randint(-30, 31, 1)[0]  # random entre 1 et 5  #-30 /+30
            aug = iaa.Affine(rotate=rotate_value)
            return rotate_value, aug

        img_aug = []
        label_transform = -2 * np.ones(10)
        if type in ['translation', 'stack']:
            label_transform[1] = 0
            translate_value_x, translate_value_y, aug = generate_translation()
            label_transform[2] = translate_value_x / self.training_size  # normalisation
            label_transform[3] = translate_value_y / self.training_size  # normalisation
            img_aug.append(aug)
        if type in ['rotation', 'stack']:
            label_transform[1] = 1
            value_transfo, aug = generate_rotation()
            label_transform[4] = value_transfo / self.training_size  # normalisation
            img_aug.append(aug)
        if type not in ['rotation', 'translation', 'stack']:
            raise NotImplementedError('The type of the unlabelised image transformation is wrong')
        if type == 'stack':
            label_transform[1] = 2
        aug = iaa.Sequential(img_aug).to_deterministic()
        image_transform = aug.augment_images([image])[0]
        label_transform = self.check_fp_in_facebox(label, label_transform, aug, image.shape)
        dim = np.insert(np.array(image_transform.shape), 0, 1)
        return image_transform.reshape(dim), label_transform

    def check_fp_in_facebox(self, labels, label_transform, aug, image_shape):
        """
        Check if the facepoint are out of the frame after the image is resize.
        In that case store this information in the label of the transformation with binary value.
        id 5 -> right eye out of the frame  6 -> left eye  7-> nose 8-> right mouse  9 -> left mouse
        :param label:
        :param label_transform:
        :param aug:
        :return:
        """

        def get_resize_augment(labels, image_shape):
            ia_face_keypoints = self.get_keypoints_on_image(image_shape, labels, type='face_box')
            w, h = image_shape[:2]
            img_aug_conf = iaa.Sequential(
                [iaa.CropAndPad(px=(-int(ia_face_keypoints.keypoints[0].y), -w + int(ia_face_keypoints.keypoints[2].x)
                                    , -h + int(ia_face_keypoints.keypoints[2].y),
                                    -int(ia_face_keypoints.keypoints[0].x))),
                 iaa.Resize(size=(self.training_size, self.training_size))])
            return img_aug_conf

        # Apply Transfo
        ia_fp_keypoints = self.get_keypoints_on_image(image_shape, labels, type='5FP')
        ia_fp_keypoints = aug.augment_keypoints([ia_fp_keypoints])[0]
        # Apply Resize
        aug_resize = get_resize_augment(labels, image_shape)
        ia_conf_det = aug_resize.to_deterministic()
        ia_fp_keypoints = ia_conf_det.augment_keypoints([ia_fp_keypoints])[0]
        # Check in facebox
        for i, keypoint in enumerate(ia_fp_keypoints.keypoints):
            if (keypoint.x > 96) or (keypoint.x < 0) or (keypoint.y > 96) or (keypoint.y < 0):
                label_transform[i + 5] = 0
            else:
                label_transform[i + 5] = 1
        return label_transform

    def augment_image_and_labels(self, image, labels, unsupervised=False):
        """
        Apply Data Augmentation (ia_conf or ia_conf_unsupervised depending on the input boolean) and cropping on the
        input image.
        :param image:
        :param labels:
        :param unsupervised:
        :return:
        """

        def _group(l, ns):
            assert len(l) == np.sum(ns)
            out = []
            for n in ns:
                out.append([l.pop(0) for _ in range(n)])
            return out

        def get_transformation_to_apply(unsupervised):
            """
            Choose the right transformation depending on input
            :param unsupervised:
            :return:
            """
            if unsupervised:
                return self.ia_conf_unsupervised.to_deterministic()
            else:
                return self.ia_conf.to_deterministic()

        def apply_transformation(ia_conf_det, image, ia_fp_keypoints, ia_face_keypoints):
            image = ia_conf_det.augment_images([image])[0]
            ia_fp_keypoints = ia_conf_det.augment_keypoints([ia_fp_keypoints])[0]
            ia_face_keypoints = ia_conf_det.augment_keypoints([ia_face_keypoints])[0]
            return image, ia_fp_keypoints, ia_face_keypoints

        def apply_resize(image, ia_fp_keypoints, ia_face_keypoints):
            # create resize transformation
            img_aug_conf = iaa.Sequential(
                [iaa.CropAndPad(px=(-int(ia_face_keypoints.keypoints[0].y), -w + int(ia_face_keypoints.keypoints[2].x)
                                    , -h + int(ia_face_keypoints.keypoints[2].y),
                                    -int(ia_face_keypoints.keypoints[0].x))),
                 iaa.Resize(size=(self.training_size, self.training_size))])
            ia_conf_det = img_aug_conf.to_deterministic()
            # Apply transformation
            image = ia_conf_det.augment_images([image])[0]
            ia_fp_keypoints = ia_conf_det.augment_keypoints([ia_fp_keypoints])[0]
            ia_face_keypoints = ia_conf_det.augment_keypoints([ia_face_keypoints])[0]
            return image, ia_fp_keypoints, ia_face_keypoints

        if len(labels) > 0:
            ia_fp_keypoints = self.get_keypoints_on_image(image.shape, labels, type='5FP')
            ia_face_keypoints = self.get_keypoints_on_image(image.shape, labels, type='face_box')
            ia_conf_det = get_transformation_to_apply(unsupervised)
            # Transform Image and Key points
            image, ia_fp_keypoints, ia_face_keypoints = apply_transformation(ia_conf_det, image, ia_fp_keypoints,
                                                                             ia_face_keypoints)
            w, h = image.shape[:2]
            # Resize imge
            image, ia_fp_keypoints, ia_face_keypoints = apply_resize(image, ia_fp_keypoints, ia_face_keypoints)

            if (self.ia_conf2 != None):
                ia_conf_det = self.ia_conf2.to_deterministic()
                image, ia_fp_keypoints, ia_face_keypoints = apply_transformation(ia_conf_det, image, ia_fp_keypoints,
                                                                                 ia_face_keypoints)
            # Reformat labels
            ia_fp_keypoints = _group(ia_fp_keypoints.keypoints, [len(obj['5FP']) for obj in labels])
            ia_face_keypoints = _group(ia_face_keypoints.keypoints, [len(obj['face_box']) for obj in labels])
            labels = [{'class': obj['class'],
                       '5FP': [(kp.x, kp.y) for kp in kps], 'face_box': [(kp.x, kp.y) for kp in kpb]}
                      for obj, kps, kpb in zip(labels, ia_fp_keypoints, ia_face_keypoints)]
        else:
            image = self.ia_conf.augment_images([image])[0]
        return image, labels

    def get_keypoints_on_image(self, image_shape, labels, type):
        """
        Retrieve KeypointsOnImage instance of face point label and face box position
        :param image:
        :param labels:
        :return:
        """

        def _flatten(l):
            return [x for xi in l for x in xi]

        keypoints = _flatten([obj[type] for obj in labels])
        ia_keypoints = ia.KeypointsOnImage(keypoints=[ia.Keypoint(*p) for p in keypoints], shape=image_shape)
        return ia_keypoints

    def on_epoch_end(self):
        if self.ia_conf is not None:
            self.ia_conf.reseed()
        if self.shuffle:
            np.random.shuffle(self.data_list)
