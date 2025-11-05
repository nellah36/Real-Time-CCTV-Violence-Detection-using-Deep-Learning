import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import models
from dataGenerator import DataGenerator 
import os
import shutil

# =========================================================================
# <<< CRITICAL PATH CONFIGURATION >>> 
# 1. Path to your RAW AVI test clip file (The load_data function will process it)
#    NOTE: I've converted the path to use forward slashes for cross-platform compatibility.
INPUT_CLIP_PATH = os.path.join('D:', 'rwf2000', 'processed', 'val', 'fight', 'XM0NI7ZmwWM_2.npy')
# 2. Path to the best saved model weights (BASE NAME ONLY - NO EXTENSION)
BEST_WEIGHTS_PATH = 'D:/fela_results/rwf2000_best_val_acc_Model' 
# =========================================================================

# --- Model Constants (Must match your training run) ---
INPUT_FRAME_SIZE = 160
VID_LEN = 32
MODE = 'both' 
LSTM_TYPE = 'sepconv' 
FUSION_TYPE = 'C' 

def predict_single_clip(clip_path):
    print("--- Starting Model Prediction ---")

    # --- 1. Load the Model Architecture ---
    model = models.getProposedModelC(
        size=INPUT_FRAME_SIZE, 
        seq_len=VID_LEN, 
        cnn_trainable=True, 
        frame_diff_interval=1, 
        mode=MODE, 
        lstm_type=LSTM_TYPE
    )
    
    # --- 2. Compile the Model (Required before load_weights) ---
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['acc'])
    
    # --- 3. Load the Saved Weights ---
    print(f"Attempting to load weights from: {BEST_WEIGHTS_PATH}")
    
    # Check if the required index file exists
    if not os.path.exists(BEST_WEIGHTS_PATH + '.index'):
        print(f"FATAL ERROR: Index file not found at {BEST_WEIGHTS_PATH}.index")
        return

    try:
        model.load_weights(BEST_WEIGHTS_PATH).expect_partial()
        print(f"Successfully loaded model weights from {BEST_WEIGHTS_PATH}")
    except Exception as e:
        print(f"FATAL ERROR during model loading: {e}")
        return

    
    # --- 4. Prepare the Input Clip using DataGenerator logic ---
    
    # Set the directory to a safe path to avoid the previous crash (D:/rwf2000)
    # The actual path being loaded is 'clip_path' (the AVI file).
    DATASET_BASE_PATH = 'D:/rwf2000'
    
    temp_generator = DataGenerator(
        directory=DATASET_BASE_PATH, # Safe dummy path for initialization
        data_augmentation=False, 
        target_frames=VID_LEN,
        resize=INPUT_FRAME_SIZE,
        normalize_=True, 
        mode=MODE,
        shuffle=False
    )
    
    # Clean the input path to prevent "embedded null character" error
    clean_clip_path = str(clip_path).strip()
    
    # Load and preprocess the data (load_data should handle AVIs if your setup is complete)
    try:
        data, diff_data = temp_generator.load_data(clean_clip_path)
    except FileNotFoundError:
        print(f"ERROR: Input clip not found at {clean_clip_path}")
        return
    except ValueError as e:
        # Re-raise the error to provide context if it's the null char issue again
        if "embedded null character" in str(e):
             print(f"FATAL ERROR: Path still contains hidden characters. Please check your INPUT_CLIP_PATH variable.")
        else:
            print(f"FATAL ERROR during data loading: {e}")
        return


    # Add the batch dimension (for a single prediction)
    data = np.expand_dims(data, axis=0) 
    diff_data = np.expand_dims(diff_data, axis=0)
    print(f"Frame stream shape: {data.shape}")
    print(f"Difference stream shape: {diff_data.shape}")

    # --- 5. Predict and Interpret ---
    prediction = model.predict([data, diff_data]) 
    # Get the probability of the positive class (Fight)
    probability_of_fight = prediction[0][0] 

    print("\n--- Prediction Result ---")
    print(f"Input Clip: {os.path.basename(clip_path)}")
    print(f"Probability of Fight: {probability_of_fight:.4f}")

    if probability_of_fight > 0.5:
        print("Classification: 🔴 FIGHT DETECTED")
    else:
        print("Classification: 🟢 NO FIGHT")
        
    print("-----------------------------------")

if __name__ == "__main__":
    predict_single_clip(INPUT_CLIP_PATH)