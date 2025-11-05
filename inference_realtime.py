import os
os.environ['PYTHONHASHSEED'] = '42'
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from collections import deque
import argparse
import models
from datetime import datetime
import json
import matplotlib.pyplot as plt

class ViolenceDetector:
    def __init__(self, weights_path, model_vidlen=32, input_size=160, 
                 lstm_type='sepconv', fusion_type='C', threshold=0.5):
        """
        Initialize Violence Detector
        
        Args:
            weights_path: Path to trained model weights
            model_vidlen: Sequence length model expects
            input_size: Frame size (160x160)
            lstm_type: Type of LSTM layer
            fusion_type: Fusion type (C/A/M)
            threshold: Classification threshold
        """
        self.model_vidlen = model_vidlen
        self.input_size = input_size
        self.threshold = threshold
        
        print(f"\n{'='*60}")
        print("INITIALIZING VIOLENCE DETECTOR")
        print(f"{'='*60}")
        print(f"Model VidLen: {model_vidlen}")
        print(f"Input Size: {input_size}x{input_size}")
        print(f"LSTM Type: {lstm_type}")
        print(f"Fusion Type: {fusion_type}")
        print(f"Threshold: {threshold}")
        print(f"{'='*60}\n")
        
        # Load model
        if fusion_type == 'C':
            model_function = models.getProposedModelC
        elif fusion_type == 'A':
            model_function = models.getProposedModelA
        elif fusion_type == 'M':
            model_function = models.getProposedModelM
        
        print("> Building model architecture...")
        self.model = model_function(
            size=input_size,
            seq_len=model_vidlen,
            frame_diff_interval=1,
            mode="both",
            lstm_type=lstm_type
        )
        
        optimizer = Adam(lr=4e-4, amsgrad=True)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
        
        print(f"> Loading weights from: {weights_path}")
        self.model.load_weights(weights_path).expect_partial()
        self.model.trainable = False
        print("✓ Model loaded successfully!\n")
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=model_vidlen)
        self.prediction_history = deque(maxlen=100)
        
    def preprocess_frame(self, frame):
        """Preprocess a single frame"""
        # Resize to input size
        frame = cv2.resize(frame, (self.input_size, self.input_size))
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def background_suppression(self, frames):
        """Apply background suppression"""
        frames = np.array(frames, dtype=np.float32)
        avg_back = np.mean(frames, axis=0)
        frames = np.abs(frames - avg_back)
        return frames
    
    def normalize(self, data):
        """Normalize data"""
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-7)
    
    def frame_difference(self, frames):
        """Calculate frame differences"""
        diffs = []
        for i in range(len(frames) - 1):
            diff = frames[i+1] - frames[i]
            diffs.append(diff)
        return np.array(diffs, dtype=np.float32)
    
    def predict(self, frame):
        """
        Predict violence for incoming frame
        
        Args:
            frame: BGR frame from video
            
        Returns:
            prediction: Violence probability (0-1)
            is_violence: Boolean (True if violence detected)
        """
        # Preprocess and add to buffer
        processed_frame = self.preprocess_frame(frame)
        self.frame_buffer.append(processed_frame)
        
        # Need full buffer for prediction
        if len(self.frame_buffer) < self.model_vidlen:
            return 0.0, False
        
        # Prepare data
        frames = np.array(list(self.frame_buffer))
        
        # Background suppression
        frames_bg = self.background_suppression(frames.copy())
        frames_bg = self.normalize(frames_bg)
        
        # Frame differences
        diffs = self.frame_difference(frames.copy())
        diffs = self.normalize(diffs)
        
        # Add batch dimension
        frames_batch = np.expand_dims(frames_bg, axis=0)
        diffs_batch = np.expand_dims(diffs, axis=0)
        
        # Predict
        prediction = self.model.predict([frames_batch, diffs_batch], verbose=0)[0][0]
        is_violence = prediction >= self.threshold
        
        # Store in history
        self.prediction_history.append(prediction)
        
        return float(prediction), bool(is_violence)
    
    def get_smoothed_prediction(self, window=5):
        """Get smoothed prediction over last N frames"""
        if len(self.prediction_history) < window:
            return np.mean(self.prediction_history) if self.prediction_history else 0.0
        return np.mean(list(self.prediction_history)[-window:])

def process_video(video_path, detector, output_path=None, show_live=True, 
                 save_results=True, smoothing_window=5):
    """
    Process video file and detect violence
    
    Args:
        video_path: Path to input video
        detector: ViolenceDetector instance
        output_path: Path to save annotated video
        show_live: Show live preview
        save_results: Save results JSON
        smoothing_window: Window for temporal smoothing
    """
    print(f"\n{'='*60}")
    print("PROCESSING VIDEO")
    print(f"{'='*60}")
    print(f"Input: {video_path}")
    if output_path:
        print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f}s\n")
    
    # Video writer
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Results storage
    results = {
        'video_path': video_path,
        'timestamp': datetime.now().isoformat(),
        'fps': fps,
        'total_frames': total_frames,
        'threshold': detector.threshold,
        'smoothing_window': smoothing_window,
        'frames': []
    }
    
    frame_idx = 0
    violence_frames = 0
    
    print("> Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get prediction
        prediction, is_violence = detector.predict(frame)
        smoothed_pred = detector.get_smoothed_prediction(smoothing_window)
        smoothed_violence = smoothed_pred >= detector.threshold
        
        if smoothed_violence:
            violence_frames += 1
        
        # Annotate frame
        annotated_frame = frame.copy()
        
        # Draw prediction bar
        bar_height = 30
        bar_y = height - bar_height - 10
        bar_width = int(width * 0.3)
        bar_x = 10
        
        # Background bar
        cv2.rectangle(annotated_frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Prediction bar (smoothed)
        pred_width = int(bar_width * smoothed_pred)
        color = (0, 0, 255) if smoothed_violence else (0, 255, 0)
        cv2.rectangle(annotated_frame, (bar_x, bar_y), 
                     (bar_x + pred_width, bar_y + bar_height), 
                     color, -1)
        
        # Text annotations
        status_text = "VIOLENCE DETECTED" if smoothed_violence else "Normal"
        status_color = (0, 0, 255) if smoothed_violence else (0, 255, 0)
        
        cv2.putText(annotated_frame, status_text, (bar_x, bar_y - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        confidence_text = f"Confidence: {smoothed_pred:.2%}"
        cv2.putText(annotated_frame, confidence_text, (bar_x, bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        frame_text = f"Frame: {frame_idx}/{total_frames}"
        cv2.putText(annotated_frame, frame_text, (width - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Save frame data
        if frame_idx % 10 == 0:  # Save every 10th frame to reduce size
            results['frames'].append({
                'frame_idx': frame_idx,
                'timestamp': frame_idx / fps,
                'prediction': float(prediction),
                'smoothed_prediction': float(smoothed_pred),
                'is_violence': bool(smoothed_violence)
            })
        
        # Write to output video
        if out:
            out.write(annotated_frame)
        
        # Show live preview
        if show_live:
            display_frame = cv2.resize(annotated_frame, (960, 540))
            cv2.imshow('Violence Detection', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n> Stopped by user")
                break
        
        frame_idx += 1
        
        # Progress
        if frame_idx % 100 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)")
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Statistics
    violence_percentage = (violence_frames / frame_idx) * 100
    results['statistics'] = {
        'frames_processed': frame_idx,
        'violence_frames': violence_frames,
        'violence_percentage': violence_percentage
    }
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Frames Processed: {frame_idx}")
    print(f"Violence Detected: {violence_frames} frames ({violence_percentage:.2f}%)")
    print(f"{'='*60}\n")
    
    # Save results
    if save_results:
        results_path = output_path.replace('.mp4', '_results.json') if output_path else 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"✓ Results saved: {results_path}\n")
    
    return results

def process_webcam(detector, smoothing_window=5):
    """
    Process webcam feed in real-time
    
    Args:
        detector: ViolenceDetector instance
        smoothing_window: Window for temporal smoothing
    """
    print(f"\n{'='*60}")
    print("REAL-TIME WEBCAM DETECTION")
    print(f"{'='*60}")
    print("Press 'q' to quit")
    print(f"{'='*60}\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Cannot open webcam")
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get prediction
        prediction, is_violence = detector.predict(frame)
        smoothed_pred = detector.get_smoothed_prediction(smoothing_window)
        smoothed_violence = smoothed_pred >= detector.threshold
        
        # Annotate frame
        height, width = frame.shape[:2]
        
        # Status text
        status_text = "VIOLENCE DETECTED!" if smoothed_violence else "Normal"
        status_color = (0, 0, 255) if smoothed_violence else (0, 255, 0)
        
        cv2.putText(frame, status_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        
        confidence_text = f"Confidence: {smoothed_pred:.2%}"
        cv2.putText(frame, confidence_text, (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Real-Time Violence Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Webcam detection stopped\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weightsPath', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--videoPath', type=str, default=None,
                       help='Path to input video file (or webcam if not provided)')
    parser.add_argument('--outputPath', type=str, default=None,
                       help='Path to save annotated output video')
    parser.add_argument('--modelVidLen', type=int, default=32,
                       help='Sequence length model expects')
    parser.add_argument('--inputSize', type=int, default=160,
                       help='Input frame size')
    parser.add_argument('--lstmType', type=str, default='sepconv',
                       choices=['sepconv', 'asepconv'])
    parser.add_argument('--fusionType', type=str, default='C',
                       choices=['C', 'A', 'M'])
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    parser.add_argument('--smoothingWindow', type=int, default=5,
                       help='Temporal smoothing window')
    parser.add_argument('--noDisplay', action='store_true',
                       help='Disable live display')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ViolenceDetector(
        weights_path=args.weightsPath,
        model_vidlen=args.modelVidLen,
        input_size=args.inputSize,
        lstm_type=args.lstmType,
        fusion_type=args.fusionType,
        threshold=args.threshold
    )
    
    # Process video or webcam
    if args.videoPath:
        output_path = args.outputPath or args.videoPath.replace('.mp4', '_annotated.mp4')
        process_video(
            video_path=args.videoPath,
            detector=detector,
            output_path=output_path,
            show_live=not args.noDisplay,
            smoothing_window=args.smoothingWindow
        )
    else:
        process_webcam(detector, smoothing_window=args.smoothingWindow)

if __name__ == "__main__":
    main()
