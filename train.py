import os
os.environ['PYTHONHASHSEED'] = '42'
from numpy.random import seed, shuffle
from random import seed as rseed
from tensorflow.random import set_seed
seed(42)
rseed(42)
set_seed(42)
import random
import pickle
import shutil
import models
from utils import *
from dataGenerator import *
from datasetProcess import *
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras import backend as K
import pandas as pd
import argparse
import tensorflow as tf

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU Memory Growth Enabled.")
    except RuntimeError as e:
        print(e)

def train(args):
    mode = args.mode
    
    if args.fusionType != 'C':
        if args.mode != 'both':
            print("Only Concat fusion supports one stream versions. Changing mode to 'both'...")
            mode = "both"
        if args.lstmType == '3dconvblock':
            raise Exception('3dconvblock instead of lstm is only available for fusionType C!')

    if args.fusionType == 'C':
        model_function = models.getProposedModelC
    elif args.fusionType == 'A':
        model_function = models.getProposedModelA
    elif args.fusionType == 'M':
        model_function = models.getProposedModelM

    dataset = args.dataset
    dataset_videos = {'hockey':'raw_videos/HockeyFights','movies':'raw_videos/movies'}

    # IMPROVED: Lower initial learning rate
    if dataset == "rwf2000":
        initial_learning_rate = 1e-04  # Reduced from 4e-04
    elif dataset == "hockey":
        initial_learning_rate = 5e-07 
    elif dataset == "movies":
        initial_learning_rate = 5e-06 

    batch_size = args.batchSize
    vid_len = args.vidLen
    
    if dataset == "rwf2000":
        dataset_frame_size = 160
    else:
        dataset_frame_size = 224
    
    frame_diff_interval = 1
    input_frame_size = 160
    lstm_type = args.lstmType

    crop_dark = {
        'hockey': (16,45),
        'movies': (18,48),
        'rwf2000': (0,0)
    }

    # IMPROVED: Increased regularization
    cnn_dropout = args.cnnDropout      # Now configurable
    lstm_dropout = args.lstmDropout    # Now configurable
    dense_dropout = args.denseDropout  # Now configurable
    weight_decay = args.weightDecay    # Now configurable

    epochs = args.numEpochs
    preprocess_data = args.preprocessData
    create_new_model = (not args.resume)
    save_path = args.savePath
    resume_path = args.resumePath
    background_suppress = args.noBackgroundSuppression

    if resume_path == "NOT_SET":
        currentModelPath = os.path.join(save_path, str(dataset) + '_currentModel')
    else:
        currentModelPath = resume_path

    bestValPath = os.path.join(save_path, str(dataset) + '_best_val_acc_Model')
    
    rwfPretrainedPath = args.rwfPretrainedPath
    if rwfPretrainedPath == "NOT_SET":
        if lstm_type == "sepconv":
            rwfPretrainedPath = "./trained_models/rwf2000_model/sepconvlstm-M/model/rwf2000_model"
        else:
            pass

    resume_learning_rate = args.resumeLearningRate
    cnn_trainable = True
    one_hot = False
    loss = 'binary_crossentropy'

    print("\n" + "="*70)
    print("IMPROVED TRAINING CONFIGURATION")
    print("="*70)
    print(f"Dataset: {dataset}")
    print(f"Batch Size: {batch_size}")
    print(f"Video Length: {vid_len} frames")
    print(f"Input Frame Size: {input_frame_size}x{input_frame_size}")
    print(f"Initial Learning Rate: {initial_learning_rate}")
    print(f"CNN Dropout: {cnn_dropout}")
    print(f"LSTM Dropout: {lstm_dropout}")
    print(f"Dense Dropout: {dense_dropout}")
    print(f"Weight Decay (L2): {weight_decay}")
    print(f"Epochs: {epochs}")
    print(f"CNN Trainable: {cnn_trainable}")
    print(f"Background Suppression: {background_suppress}")
    print(f"Fusion Type: {args.fusionType}")
    print(f"LSTM Type: {lstm_type}")
    print("="*70 + "\n")

    if preprocess_data:
        if dataset == 'rwf2000':
            RWF_INPUT_PATH = 'D:/rwf2000/RWF-2000/'
            RWF_OUTPUT_PATH = 'D:/rwf2000/processed/'
            convert_dataset_to_npy(src=RWF_INPUT_PATH, dest=RWF_OUTPUT_PATH, 
                                   crop_x_y=None, target_frames=vid_len, 
                                   frame_size=dataset_frame_size)
        else:
            if os.path.exists('{}'.format(dataset)):
                shutil.rmtree('{}'.format(dataset))
            split = train_test_split(dataset_name=dataset, source=dataset_videos[dataset])
            os.mkdir(dataset)
            os.mkdir(os.path.join(dataset,'videos'))
            move_train_test(dest='{}/videos'.format(dataset), data=split)
            os.mkdir(os.path.join(dataset,'processed'))
            convert_dataset_to_npy(src='{}/videos'.format(dataset),
                                   dest='{}/processed'.format(dataset), 
                                   crop_x_y=crop_dark[dataset], 
                                   target_frames=vid_len, 
                                   frame_size=dataset_frame_size)
    
    if dataset == 'rwf2000':
        data_generator_base_path = 'D:/rwf2000'
        val_dir = 'val' 
    else:
        data_generator_base_path = dataset 
        val_dir = 'test' 
        
    train_generator = DataGenerator(
        directory='{}/processed/train'.format(data_generator_base_path),
        batch_size=batch_size,
        data_augmentation=True,
        shuffle=True,
        one_hot=one_hot,
        sample=False,
        resize=input_frame_size,
        background_suppress=background_suppress,
        target_frames=vid_len,
        dataset=dataset,
        mode=mode
    )

    test_generator = DataGenerator(
        directory='{}/processed/{}'.format(data_generator_base_path, val_dir),
        batch_size=batch_size,
        data_augmentation=False,
        shuffle=False,
        one_hot=one_hot,
        sample=False,
        resize=input_frame_size,
        background_suppress=background_suppress,
        target_frames=vid_len,
        dataset=dataset,
        mode=mode
    )

    print('> CNN Trainable:', cnn_trainable)
    
    if create_new_model:
        print('> Creating new model with improved regularization...')
        model = model_function(
            size=input_frame_size, 
            seq_len=vid_len,
            cnn_trainable=cnn_trainable, 
            frame_diff_interval=frame_diff_interval, 
            mode=mode, 
            lstm_type=lstm_type,
            cnn_dropout=cnn_dropout,
            lstm_dropout=lstm_dropout,
            dense_dropout=dense_dropout,
            weight_decay=weight_decay
        )
        
        if dataset == "hockey" or dataset == "movies":
            print('> Loading weights pretrained on RWF dataset from', rwfPretrainedPath)
            model.load_weights(rwfPretrainedPath)
        
        optimizer = Adam(lr=initial_learning_rate, amsgrad=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
        print('> New model created') 
    else:
        print('> Resuming from checkpoint:', currentModelPath)  
        model = model_function(
            size=input_frame_size, 
            seq_len=vid_len,
            cnn_trainable=cnn_trainable, 
            frame_diff_interval=frame_diff_interval, 
            mode=mode, 
            lstm_type=lstm_type,
            cnn_dropout=cnn_dropout,
            lstm_dropout=lstm_dropout,
            dense_dropout=dense_dropout,
            weight_decay=weight_decay
        )
        optimizer = Adam(lr=resume_learning_rate, amsgrad=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
        model.load_weights(currentModelPath)

    print('> Model Summary:')
    model.summary(line_length=140)
    print('> Optimizer Config:', model.optimizer.get_config())

    dot_img_file = os.path.join(save_path, 'model_architecture.png')
    print('> Plotting model architecture to:', dot_img_file)
    plot_model(model, to_file=dot_img_file, show_shapes=True)

    # IMPROVED: Better callbacks
    modelcheckpoint = ModelCheckpoint(
        currentModelPath, 
        monitor='loss', 
        verbose=1, 
        save_best_only=False, 
        save_weights_only=True, 
        mode='auto', 
        save_freq='epoch'
    )
    
    modelcheckpointVal = ModelCheckpoint(
        bestValPath, 
        monitor='val_acc', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=True, 
        mode='auto', 
        save_freq='epoch'
    )

    # IMPROVED: Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    # IMPROVED: Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=args.reduceLRPatience,
        verbose=1,
        mode='min',
        min_lr=1e-7
    )

    historySavePath = os.path.join(save_path, 'results', str(dataset))
    save_training_history = SaveTrainingCurves(save_path=historySavePath)

    callback_list = [
        modelcheckpoint,
        modelcheckpointVal,
        early_stopping,
        reduce_lr,
        save_training_history
    ]
    
    if args.useLRScheduler:
        callback_list.append(LearningRateScheduler(lr_scheduler, verbose=1))

    print("\n> Starting training...")
    print(f"> Training samples: {len(train_generator) * batch_size}")
    print(f"> Validation samples: {len(test_generator) * batch_size}")
    print(f"> Steps per epoch: {len(train_generator)}\n")

    history = model.fit(
        steps_per_epoch=len(train_generator),
        x=train_generator,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator),
        verbose=1,
        workers=8,
        max_queue_size=8,
        use_multiprocessing=False,
        callbacks=callback_list
    )

    # Save training history
    history_path = os.path.join(save_path, f'{dataset}_training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"\n> Training history saved to: {history_path}")

    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best model saved at: {bestValPath}")
    print(f"Latest checkpoint: {currentModelPath}")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numEpochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--vidLen', type=int, default=32, help='Number of frames in a clip')
    parser.add_argument('--batchSize', type=int, default=4, help='Training batch size')
    parser.add_argument('--resume', help='Resume from previous checkpoint', action='store_true')
    parser.add_argument('--noBackgroundSuppression', help='Use background suppression', action='store_false')
    parser.add_argument('--preprocessData', help='Preprocess data to NPY', action='store_true')
    
    # Model architecture
    parser.add_argument('--mode', type=str, default='both', 
                       choices=['both', 'only_frames', 'only_differences']) 
    parser.add_argument('--dataset', type=str, default='rwf2000', 
                       choices=['rwf2000','movies','hockey']) 
    parser.add_argument('--lstmType', type=str, default='sepconv', 
                       choices=['sepconv','asepconv', 'conv', '3dconvblock'])
    parser.add_argument('--fusionType', type=str, default='C', 
                       choices=['C','A','M']) 
    
    # IMPROVED: Configurable regularization
    parser.add_argument('--cnnDropout', type=float, default=0.35, 
                       help='CNN dropout rate (default: 0.35, was 0.25)')
    parser.add_argument('--lstmDropout', type=float, default=0.35, 
                       help='LSTM dropout rate (default: 0.35, was 0.25)')
    parser.add_argument('--denseDropout', type=float, default=0.5, 
                       help='Dense layer dropout rate (default: 0.5, was 0.3)')
    parser.add_argument('--weightDecay', type=float, default=1e-4, 
                       help='L2 weight decay (default: 1e-4, was 2e-5)')
    
    # Callbacks
    parser.add_argument('--patience', type=int, default=10, 
                       help='Early stopping patience (epochs)')
    parser.add_argument('--reduceLRPatience', type=int, default=5, 
                       help='ReduceLROnPlateau patience (epochs)')
    parser.add_argument('--useLRScheduler', action='store_true',
                       help='Use custom learning rate scheduler')
    
    # Paths
    parser.add_argument('--savePath', type=str, default='D:/fela_results', 
                       help='Folder to save models')
    parser.add_argument('--rwfPretrainedPath', type=str, default='NOT_SET', 
                       help='Path to RWF pretrained weights')
    parser.add_argument('--resumePath', type=str, default='NOT_SET', 
                       help='Path to checkpoint for resuming')
    parser.add_argument('--resumeLearningRate', type=float, default=5e-05, 
                       help='Learning rate when resuming')
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
