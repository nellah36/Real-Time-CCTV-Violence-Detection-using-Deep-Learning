import os
os.environ['PYTHONHASHSEED'] = '42'
from numpy.random import seed, shuffle
from random import seed as rseed
from tensorflow.random import set_seed
seed(42)
rseed(42)
set_seed(42)
import tensorflow as tf
import random
import pickle
import shutil
import models
from utils import *
from dataGenerator import *
from datasetProcess import *
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras import backend as K
import pandas as pd
import argparse
from tensorflow.keras.optimizers import RMSprop, Adam

def evaluate(args):

    mode = args.mode # ["both","only_frames","only_differences"]

    if args.fusionType != 'C':
        if args.mode != 'both':
            print("Only Concat fusion supports one stream versions. Changing mode to 'both'...")
            mode = "both"
        if args.lstmType == '3dconvblock':
            raise Exception('3dconvblock instead of lstm is only available for fusionType C ! aborting execution...')

    if args.fusionType == 'C':
        model_function = models.getProposedModelC
    elif args.fusionType == 'A':
        model_function = models.getProposedModelA
    elif args.fusionType == 'M':
        model_function = models.getProposedModelM

    dataset = args.dataset # ['rwf2000','movies','hockey']
    dataset_videos = {'hockey':'raw_videos/HockeyFights','movies':'raw_videos/movies'}

    batch_size = args.batchSize

    # CRITICAL FIX: Use modelVidLen for BOTH model and data generator
    vid_len = args.modelVidLen  # This is the sequence length used during training (e.g., 8)
    
    if dataset == "rwf2000":
        dataset_frame_size = 160  # Match training preprocessing
    else:
        dataset_frame_size = 224
    frame_diff_interval = 1
    
    # CRITICAL FIX: Must match training configuration (160x160, NOT 96x96)
    input_frame_size = 160  # Changed from 96 to 160 to match train.py

    lstm_type = args.lstmType

    crop_dark = {
        'hockey' : (16,45),
        'movies' : (18,48),
        'rwf2000': (0,0)
    }

    #---------------------------------------------------

    preprocess_data = args.preprocessData

    weightsPath = args.weightsPath
    if weightsPath == "NOT_SET":
        raise Exception("weights not provided!")

    one_hot = False

    #----------------------------------------------------

    if preprocess_data:
        if dataset == 'rwf2000':
            os.mkdir(os.path.join(dataset, 'processed'))
            convert_dataset_to_npy(src='{}/RWF-2000'.format(dataset), dest='{}/processed'.format(
                dataset), crop_x_y=None, target_frames=vid_len, frame_size=dataset_frame_size)
        else:
            if os.path.exists('{}'.format(dataset)):
                shutil.rmtree('{}'.format(dataset))
            split = train_test_split(dataset_name=dataset,source=dataset_videos[dataset])
            os.mkdir(dataset)
            os.mkdir(os.path.join(dataset,'videos'))
            move_train_test(dest='{}/videos'.format(dataset),data=split)
            os.mkdir(os.path.join(dataset,'processed'))
            convert_dataset_to_npy(src='{}/videos'.format(dataset),dest='{}/processed'.format(dataset), crop_x_y=crop_dark[dataset], target_frames=vid_len, frame_size=dataset_frame_size)

    # --- Determining the test data directory: Using 'val' as confirmed by user ---
    if args.dataPath != 'NOT_SET':
        test_data_dir = os.path.join(args.dataPath, 'val')
    else:
        test_data_dir = '{}/processed/val'.format(dataset) 
    
    print(f"\n{'='*60}")
    print(f"EVALUATION CONFIGURATION:")
    print(f"{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Test data directory: {test_data_dir}")
    print(f"Input frame size: {input_frame_size}x{input_frame_size}")
    print(f"Sequence length (modelVidLen): {vid_len} frames")
    print(f"Batch size: {batch_size}")
    print(f"Fusion type: {args.fusionType}")
    print(f"LSTM type: {lstm_type}")
    print(f"Mode: {mode}")
    print(f"{'='*60}\n")
    
    # CRITICAL FIX: DataGenerator must sample vid_len frames from the 32-frame clips
    test_generator = DataGenerator(directory=test_data_dir,
                                   batch_size=batch_size,
                                   data_augmentation=False,
                                   shuffle=False,
                                   one_hot=one_hot,
                                   sample=True,  # CHANGED: Enable sampling
                                   resize=input_frame_size,  # Must be 160
                                   target_frames=vid_len,  # CHANGED: Sample to modelVidLen
                                   background_suppress=True,
                                   dataset=dataset,
                                   mode=mode)
    
    # ----------------------------------------------------------------

    print('> Building model architecture...') 
    
    # CRITICAL: Model architecture must match training exactly
    model = model_function(size=input_frame_size, 
                          seq_len=vid_len,  # Use modelVidLen here
                          frame_diff_interval=frame_diff_interval, 
                          mode=mode, 
                          lstm_type=lstm_type)
    
    optimizer = Adam(lr=4e-4, amsgrad=True)
    loss = 'binary_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    
    print('> Model Summary:')
    model.summary(line_length=140)
    
    # CRITICAL: Path to the weights file
    best_model_path = os.path.join(weightsPath, 'rwf2000_best_val_acc_Model')
    
    print(f'\n> Loading weights from: {best_model_path}')
    
    # Load weights with expect_partial() to ignore optimizer state
    try:
        model.load_weights(best_model_path).expect_partial()
        print('> ✓ Weights loaded successfully!')
    except Exception as e:
        print(f'> ✗ Error loading weights: {e}')
        raise
    
    model.trainable = False
                    
    #--------------------------------------------------

    print(f'\n> Starting evaluation on {len(test_generator)} batches...\n')
    
    test_results = model.evaluate(
        steps=len(test_generator),
        x=test_generator,
        verbose=1,
        workers=8,
        max_queue_size=8,
        use_multiprocessing=False,
    )
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Test Loss:     {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
    print("="*60 + "\n")
    
    # Optional: Save results to CSV
    # save_as_csv(test_results, "", 'test_results.csv')

    #---------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelVidLen', type=int, default=8, 
                       help='Number of frames the model was trained with (CRITICAL: must match training)')
    parser.add_argument('--batchSize', type=int, default=16, 
                       help='Evaluation batch size')
    parser.add_argument('--preprocessData', 
                       help='Whether to preprocess data (make npy file from video clips)',
                       action='store_true')
    parser.add_argument('--mode', type=str, default='both', 
                       help='Model type - both, only_frames, only_differences', 
                       choices=['both', 'only_frames', 'only_differences']) 
    parser.add_argument('--dataset', type=str, default='rwf2000', 
                       help='Dataset - rwf2000, movies, hockey', 
                       choices=['rwf2000','movies','hockey']) 
    parser.add_argument('--lstmType', type=str, default='sepconv', 
                       help='LSTM type - sepconv, asepconv', 
                       choices=['sepconv','asepconv']) 
    parser.add_argument('--dataPath', type=str, default='NOT_SET', 
                       help='Path to the preprocessed data folder (e.g., D:/rwf2000/processed)') 
    parser.add_argument('--weightsPath', type=str, default='NOT_SET', 
                       help='Path to the folder containing trained weights') 
    parser.add_argument('--fusionType', type=str, default='C', 
                       help='Fusion type - A for add, M for multiply, C for concat', 
                       choices=['C','A','M']) 
    
    args = parser.parse_args()
    
    # Validation
    if args.weightsPath == "NOT_SET":
        parser.error("--weightsPath is required!")
    if args.dataPath == "NOT_SET":
        parser.error("--dataPath is required!")
    
    evaluate(args)

if __name__ == "__main__":
    main()
