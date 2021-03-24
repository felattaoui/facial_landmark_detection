import cv2
import numpy as np
import os
import imgaug as ia
from imgaug import augmenters as iaa
from tensorflow.keras.utils import Sequence
import os.path


class BatchGeneratorSupervised(Sequence):
    def __init__(self, batch_size, training_size, data_list, classes, shuffle=True, image_normalization_fn=None,
                 label_normalization_fn=None, img_aug_conf=None,
                 encoding_fn=None, preprocess_pred=False):
        self.batch_size = batch_size
        self.training_size = training_size
        self.data_list = data_list
        self.classes = classes
        self.image_normalization_fn = image_normalization_fn
        self.label_normalization_fn = label_normalization_fn
        self.ia_conf = img_aug_conf
        self.encoding_fn = encoding_fn
        self.shuffle = shuffle
        self.seeded = False

    def __len__(self):
        return int(np.ceil(self.size() / self.batch_size))

    def num_classes(self):
        return len(self.labels)

    def size(self):
        return len(self.data_list)

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

    def augment_image_and_labels(self, image, labels):
        def _flatten(l):
            return [x for xi in l for x in xi]

        def _group(l, ns):
            assert len(l) == np.sum(ns)
            out = []
            for n in ns:
                out.append([l.pop(0) for _ in range(n)])
            return out

        if len(labels) > 0:
            fp_keypoints = _flatten([obj['5FP'] for obj in labels])
            face_keypoints = _flatten([obj['face_box'] for obj in labels])
            ia_fp_keypoints = ia.KeypointsOnImage(keypoints=[ia.Keypoint(*p) for p in fp_keypoints], shape=image.shape)
            ia_face_keypoints = ia.KeypointsOnImage(keypoints=[ia.Keypoint(*p) for p in face_keypoints],
                                                    shape=image.shape)

            # to_deterministic() removes the randomness from all augmenters and makes them deterministic (e.g. for each
            # parameter that comes from a distribution, it samples one value from that distribution and then keeps
            # reusing that value). That is useful for landmark augmentation, because you want to transform images and
            # their landmarks in the same way, e.g. rotate an image and its landmarks by 30 degrees.
            # If you don't have landmarks, then you most likely don't need the function.

            ia_conf_det = self.ia_conf.to_deterministic()
            image = ia_conf_det.augment_images([image])[0]
            ia_fp_keypoints = ia_conf_det.augment_keypoints([ia_fp_keypoints])[0]
            ia_face_keypoints = ia_conf_det.augment_keypoints([ia_face_keypoints])[0]
            w, h = image.shape[:2]
            # Transformé ici
            img_aug_conf = iaa.Sequential(
                [iaa.CropAndPad(px=(-int(ia_face_keypoints.keypoints[0].y), -w + int(ia_face_keypoints.keypoints[2].x)
                                    , -h + int(ia_face_keypoints.keypoints[2].y),
                                    -int(ia_face_keypoints.keypoints[0].x))),
                 iaa.Resize(size=(self.training_size, self.training_size))])

            ia_conf_det = img_aug_conf.to_deterministic()
            image = ia_conf_det.augment_images([image])[0]  # l'image passe en 96*96
            ia_fp_keypoints = ia_conf_det.augment_keypoints([ia_fp_keypoints])[0]
            ia_face_keypoints = ia_conf_det.augment_keypoints([ia_face_keypoints])[0]

            ia_fp_keypoints = _group(ia_fp_keypoints.keypoints, [len(obj['5FP']) for obj in labels])
            ia_face_keypoints = _group(ia_face_keypoints.keypoints, [len(obj['face_box']) for obj in labels])
            labels = [{'class': obj['class'],
                       '5FP': [(kp.x, kp.y) for kp in kps], 'face_box': [(kp.x, kp.y) for kp in kpb]}
                      for obj, kps, kpb in zip(labels, ia_fp_keypoints, ia_face_keypoints)]
        else:
            image = self.ia_conf.augment_images([image])[0]
        return image, labels

    def __getitem__(self, idx):
        if not self.seeded:
            seed = int.from_bytes(os.urandom(4), byteorder='little')
            ia.seed(seed)
            if self.ia_conf is not None: self.ia_conf.reseed(seed)  # Farid added if self.ia_conf is not None
            self.seeded = True
        sub_dl = self.data_list[idx * self.batch_size:(idx + 1) * self.batch_size:]
        images = self.load_images(self, sub_dl)
        labels = self.load_labels(sub_dl)
        if self.ia_conf is not None:
            for i in range(len(images)):
                images[i], labels[i] = self.augment_image_and_labels(images[i], labels[i])
        if self.image_normalization_fn is not None:
            images = [self.image_normalization_fn(img) for img in images]
        if self.encoding_fn is not None:
            for i in range(len(images)):
                images[i], labels[i] = self.encoding_fn(images[i], labels[i])
            images = np.array(images)
            labels = np.array(labels)
        if self.label_normalization_fn is not None:
            labels = self.label_normalization_fn(labels, self.training_size)
        return images, labels

    def on_epoch_end(self):
        if self.ia_conf is not None:
            self.ia_conf.reseed()
        if self.shuffle:
            np.random.shuffle(self.data_list)
