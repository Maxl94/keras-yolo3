"""
Retrain the YOLO model for your own dataset.
"""

import datetime
import os
import time

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import Sequence

from yolo3.model import yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data, preprocess_true_boxes


def _main():
    n_epochs = None
    initial_epoch = None
    train_mode = 'all'

    assert train_mode in ['all', 'complete', 'frozen'], 'Training mode must be "all", "complete" or "frozen"'

    log_prefix = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_test'

    workers = 20

    annotation_path = 'data/labels_real/yolo_labels.txt'
    validation_path = 'data/labels_real/val_yolo_labels.txt'

    batch_size_frozen = 8
    batch_size_complete = 4

    epochs_frozen = 10
    epochs_complete = 10

    classes_path = 'model_data/ma_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (608, 608)  # multiple of 32, hw

    with open(annotation_path) as f:
        lines = f.readlines()

    with open(validation_path) as f:
        validation_lines = f.readlines()

    num_val = len(validation_lines)
    num_train = len(lines)

    assert num_train % max(batch_size_frozen, batch_size_complete) == 0, f'Error: Number of training samples ({num_train}) must dividable ' \
        f'by batch size ({max(batch_size_frozen, batch_size_complete)})'
    assert num_val % max(batch_size_frozen, batch_size_complete) == 0, f'Error: Number of validation samples ({num_val}) must dividable ' \
        f'by batch size ({max(batch_size_frozen, batch_size_complete)})'

    log_dir = f'logs/{log_prefix}_{os.path.basename(anchors_path)[:-4]}_bsf_{batch_size_frozen}_bfc_{batch_size_complete}'

    is_tiny_version = len(anchors) == 6  # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2, weights_path='model_data/yolo_weights.h5')  # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    start_time = time.time()

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if train_mode == 'all' or train_mode == 'frozen':
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = batch_size_frozen
        data_gen_train = DataGenerator(lines, batch_size, input_shape, anchors, num_classes)
        data_gen_validation = DataGenerator(validation_lines, batch_size, input_shape, anchors, num_classes)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        r = model.fit_generator(data_gen_train,
                                steps_per_epoch=max(1, num_train // batch_size),
                                validation_data=data_gen_validation,
                                validation_steps=max(1, num_val // batch_size),
                                epochs=epochs_frozen,
                                initial_epoch=0,
                                callbacks=[logging, checkpoint],
                                workers=workers,
                                max_queue_size=20)
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')
        n_epochs = len(r.history['loss'])

        del data_gen_train
        del data_gen_validation

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if train_mode == 'all' or train_mode == 'complete':
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.')

        initial_epoch = n_epochs if n_epochs is not None else 0
        epochs = initial_epoch + epochs_complete
        batch_size = batch_size_complete  # note that more GPU memory is required after unfreezing the body

        data_gen_train = DataGenerator(lines, batch_size, input_shape, anchors, num_classes)
        data_gen_validation = DataGenerator(validation_lines, batch_size, input_shape, anchors, num_classes)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        r = model.fit_generator(data_gen_train,
                                steps_per_epoch=max(1, num_train // batch_size),
                                validation_data=data_gen_validation,
                                validation_steps=max(1, num_val // batch_size),
                                epochs=epochs,
                                initial_epoch=initial_epoch,
                                callbacks=[logging, checkpoint, reduce_lr, early_stopping],
                                workers=workers,
                                max_queue_size=20)
        model.save_weights(log_dir + 'trained_weights_final.h5')
        n_epochs = len(r.history['loss'])

    # Further training if needed.
    end_time = time.time()
    delta = float(end_time - start_time)
    print(f'Training took {delta:.1f}s')
    print(f'With an average time per epoch of {delta / (n_epochs + initial_epoch if initial_epoch is not None else 1):.1f}s')


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l], \
                           num_anchors // 2, num_classes + 5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers) - 2)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


class DataGenerator(Sequence):
    def __init__(self, annotation_lines, batch_size, input_shape, anchors, num_classes, max_boxes=80):
        self.annotations_lines = annotation_lines
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_classes = num_classes
        self.max_boxes = max_boxes

    def __len__(self):
        return int(np.ceil(len(self.annotations_lines) / float(self.batch_size)))

    def __getitem__(self, idx):
        annotation_lines = self.annotations_lines[idx * self.batch_size:(idx + 1) * self.batch_size]

        image_data = []
        box_data = []
        for annotation_line in annotation_lines:
            image, box = get_random_data(annotation_line, self.input_shape, random=True, max_boxes=self.max_boxes)
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
        return [image_data, *y_true], np.zeros(self.batch_size)


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    _main()
