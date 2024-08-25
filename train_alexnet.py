import os
from time import time
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from alexnet_utils.params import parser, print_arguments
from alexnet_utils.alexnet import AlexNet
import wandb
from wandb.keras import WandbCallback

class SaveHistoryCallback(Callback):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.history = {'loss': [], 'val_loss': [], 'auc_pr':[], 'val_auc_pr':[], 'val_precision':[], 'val_recall':[]}

    def on_epoch_end(self, epoch, logs=None):
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['auc_pr'].append(logs.get('auc_pr'))
        self.history['val_auc_pr'].append(logs.get('val_auc_pr'))
        self.history['val_precision'].append(logs.get('val_precision'))
        self.history['val_recall'].append(logs.get('val_recall'))

        with open(self.file_path, 'w') as f:
            json.dump(self.history, f)

def get_output_dir(output):
    # create a separate output directory for each run
    pid = os.getpid()
    outdir = os.path.join(output,f"{pid}")

    return pid, outdir

def random_choice(x, size, seed, axis=0, unique=True):
    dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
    indices = tf.range(0, dim_x, dtype=tf.int64)
    sample_index = tf.random.shuffle(indices,seed=seed)[:size]
    sample = tf.gather(x, sample_index, axis=axis)

    return sample, sample_index

def random_int_rot_img(inputs,seed):
    angles = tf.constant([1, 2, 3, 4])
    # Make a new seed.
    #new_seed = tf.random.experimental.stateless_split((seed,seed), num=1)[0, :]
    angle = random_choice(angles,1,seed=seed)[0][0]
    inputs = tf.image.rot90(inputs, k=angle)

    return inputs

def rescale(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 255.0)

    return image, label

# define custom augmentations
def augment_custom(images, labels, augmentation_types, seed):
    
    images, labels = rescale(images, labels)
    # Make a new seed.
    #new_seed = tf.random.experimental.stateless_split((seed,seed), num=1)[0, :]
    new_seed = seed
    if 'rotation' in augmentation_types:
        images = random_int_rot_img(images,seed=seed)
    if 'flip' in augmentation_types:
        images = tf.image.random_flip_left_right(images, seed=new_seed)
        images = tf.image.random_flip_up_down(images, seed=new_seed)
    if 'brightness' in augmentation_types:
        images = tf.image.random_brightness(images, max_delta=0.2, seed=new_seed)
    if 'contrast' in augmentation_types:
        images = tf.image.random_contrast(images, lower=0.2, upper=0.5, seed=new_seed)

    return (images, labels)

def get_train_data(data_dir, val_dir, train_frac, target_size, batch_size, augmentation_types, outdir, random_state):
    if val_dir is None:
        train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
          data_dir,
          validation_split=1-train_frac,
          subset="both",
          color_mode='rgb',
          seed=random_state,
          image_size=target_size,
          batch_size=None)
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
          data_dir,
          color_mode='rgb',
          seed=random_state,
          image_size=target_size,
          batch_size=None)

        val_ds = tf.keras.utils.image_dataset_from_directory(
              val_dir,
              color_mode='rgb',
              seed=random_state,
              image_size=target_size,
              batch_size=None)

    class_names = train_ds.class_names
    print("Training dataset class names are :",class_names)
    # set the num_classes automatically from the number of directories found  by the data generator
    #num_classes = len(class_names)

    train_ds_raw, val_ds_raw = train_ds, val_ds

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
            train_ds
            .shuffle(1000)
            .map(lambda x, y: augment_custom(x, y, augmentation_types, seed=random_state), num_parallel_calls=AUTOTUNE)
            #.cache()
            .batch(batch_size)
            .prefetch(buffer_size=AUTOTUNE)
            )

    val_ds = (
            val_ds
            .map(rescale, num_parallel_calls=AUTOTUNE)
            #.cache()
            .batch(batch_size)
            .prefetch(buffer_size=AUTOTUNE)
            )

    return train_ds_raw, val_ds_raw, train_ds, val_ds

def save_filepaths(train_ds, val_ds, outdir):
    train_filenames = train_ds.file_paths
    val_filenames = val_ds.file_paths

    pd.DataFrame({"Filename":train_filenames}).to_csv(os.path.join(outdir,f"train_filenames.csv"),index=False)
    pd.DataFrame({"Filename":val_filenames}).to_csv(os.path.join(outdir,f"validation_filenames.csv"),index=False)

    # save filenames also along with images and labels in the saved dataset
    def change_inputs(images, labels, paths):
      return images, labels,  tf.constant(paths)

    # save validation data for future evaluation
    val_ds_todisk = val_ds.map(lambda images, labels: change_inputs(images, labels, paths=val_filenames))

    path = os.path.join(outdir,"val_data")
    tf.data.Dataset.save(val_ds_todisk, path)

def calc_bias(train_ds):
    # calculate an intial bias to apply based on the class imbalance in the training data
    pos = train_ds.map(lambda _, label: tf.reduce_sum(label)).reduce(0, lambda count, val: count + val).numpy()
    neg = train_ds.map(lambda _, label: tf.reduce_sum(1 - label)).reduce(0, lambda count, val: count + val).numpy()
    initial_bias = np.log([pos/neg])
    print("[INFO] Calculated initial weight bias:", initial_bias)
    return initial_bias

def get_compiled_model():
    classification_threshold = 0.5

    METRICS = [
          tf.keras.metrics.Precision(thresholds=classification_threshold,
                                     name='precision'),
          tf.keras.metrics.Recall(thresholds=classification_threshold,
                                  name="recall"),
          tf.keras.metrics.AUC(num_thresholds=100, curve='PR', name='auc_pr'),
    ]

    model = AlexNet.build(width=target_size[0], height=target_size[1], depth=channels, classes=1, reg=0.0002)


    print("[INFO] compiling model...")
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=METRICS)
    return model

if __name__=="__main__":
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)

    parser.add_argument('-images', required=True, help="path containing images of two classes")
    parser.add_argument('-epochs', required=True, type=int, default=50, help="num epochs")
    parser.add_argument('-model_path', default=None, help="Filepath to save model during training and to load model from when testing")
    parser.add_argument('-val_dir', default=None, help="path containing validation data")
    parser.add_argument('-retrain', type=bool, default=False, help="Whether to continue previous training")
    args = parser.parse_args()
    print_arguments(parser,args)

    data_dir = args.images
    target_size = args.target_size
    batch_size = args.batch_size
    train_frac = args.train_frac
    random_state = args.random_state
    num_classes = args.num_classes
    channels = args.channels
    output = args.output_dir
    epochs = args.epochs
    model_path = args.model_path
    augmentation_types = args.augmentation_types
    val_dir = args.val_dir

    # set the color_mode from the number of channels

    #color_dict = {1:'grayscale',3:'rgb'}
    #color_mode = color_dict[channels]

    # define an output directory to hold saved model, training graphs etc.
    wandb.init(project="Ring_Train")
    wandb_dir = wandb.run.name
    outdir = os.path.join("output", wandb_dir)
    print(f"Outputs are saved to {outdir}")

    if args.retrain:

        if os.path.exists(model_path):
            print("[INFO] Loading existing model from disk ..")
            model = load_model(model_path)
        else:
            print("Invalid model path....")
    else:
        model = get_compiled_model()

    if model_path is None:
        model_path = os.path.join(outdir,"best_model.h5")

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    train_ds_raw, val_ds_raw , train_ds, val_ds = get_train_data(data_dir, val_dir, train_frac, target_size, batch_size, \
            augmentation_types, outdir, random_state)

    print(f"Saving filenames used for training and validation to disk...")
    save_filepaths(train_ds_raw, val_ds_raw, outdir)

    # define callbacks
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    mc = ModelCheckpoint(model_path, monitor='val_loss', \
            mode='min', verbose=1, save_best_only=True)
    history_path = os.path.join(outdir,'history.json')
    hc = SaveHistoryCallback(history_path)

    start = time()

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, shuffle=True, callbacks=[mc, hc, tensorboard, WandbCallback(save_model=(False),save_graph=(False))])

    print("Total time taken for training: %d seconds" % (time()-start))
    wandb.finish()
