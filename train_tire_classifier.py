# train_tire_classifier.py
import pathlib, tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import EfficientNetB0

############################
# 1.  BASIC CONFIGURATION  #
############################
DATA_ROOT   = pathlib.Path("tires_images")
CLASS_DIRS  = {                # folder → class label
    "normal_tires_images":           "normal",
    "augmented_images_normal_tires": "normal",  # comment out if you don’t have this
    "snow_tires_images":             "winter",
    "snow_tires_img_with_augment":   "winter",
}
BATCH_SIZE      = 32
IMG_SIZE        = (224, 224)         # EfficientNet-B0 default
VALID_SPLIT     = 0.2
SEED            = 123
EPOCHS          = 20
OUTPUT_DIR      = pathlib.Path("tire_model_out")
OUTPUT_DIR.mkdir(exist_ok=True)

##############################################
# 2.  MERGE ALL CLASS FOLDERS INTO ONE TREE  #
##############################################
# Keras' `image_dataset_from_directory` wants one root with sub-folders per class
SYMLINK_ROOT = OUTPUT_DIR / "symlink_dataset"
if not SYMLINK_ROOT.exists():
    for cls_folder, label in CLASS_DIRS.items():
        src = DATA_ROOT / cls_folder
        dst = SYMLINK_ROOT / label / cls_folder            # keep original name to avoid clashes
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Create lightweight symlinks so we don't duplicate images
        for img in src.glob("*"):
            dst_link = dst / img.name
            if not dst_link.exists():
                dst_link.symlink_to(img.resolve())

################################
# 3.  BUILD DATASET PIPELINES  #
################################
train_ds = tf.keras.utils.image_dataset_from_directory(
    SYMLINK_ROOT,
    validation_split=VALID_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    SYMLINK_ROOT,
    validation_split=VALID_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# Prefetch for speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

#########################
# 4.  DATA AUGMENTATION #
#########################
data_augmentation = tf.keras.Sequential([
    layers.Rescaling(scale=1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

################################
# 5.  DEFINE THE EFFICIENTNET  #
################################
base_model = EfficientNetB0(
    include_top=False,
    input_shape=IMG_SIZE + (3,),
    weights="imagenet",
    pooling="avg",
)
base_model.trainable = False          # first, train only the head

inputs  = layers.Input(shape=IMG_SIZE + (3,))
x       = data_augmentation(inputs)
x       = base_model(x, training=False)
outputs = layers.Dense(1, activation="sigmoid")(x)   # binary classification

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

##################################
# 6.  TRAIN THE CLASSIFICATION   #
##################################
ckpt_cb = callbacks.ModelCheckpoint(
    OUTPUT_DIR / "best_unfrozen.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

early_cb = callbacks.EarlyStopping(
    patience=5, restore_best_weights=True
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[ckpt_cb, early_cb],
)

#############################################
# 7.  OPTIONAL:  FINE-TUNE THE BASE MODEL   #
#############################################
# Unfreeze top layers of EfficientNet for a few more epochs
base_model.trainable = True
# Re-compile with a lower learning-rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

fine_tune_epochs = 5
model.fit(
    train_ds,
    epochs=history.epoch[-1] + fine_tune_epochs + 1,
    initial_epoch=history.epoch[-1] + 1,
    validation_data=val_ds,
    callbacks=[ckpt_cb, early_cb],
)

#####################
# 8.  SAVE & TEST   #
#####################
final_path = OUTPUT_DIR / "tire_classifier.h5"
model.save(final_path)
print(f"✓ Model saved to {final_path.resolve()}")
