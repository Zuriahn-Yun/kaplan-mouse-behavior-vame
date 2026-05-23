import os
import deeplabcut

# 1. Windows Paths
video_path = [r"C:\Users\zuria\Kaplan\VAME\Subject 2 - female GO Vehicle OFT.mp4"]
video_dir = os.path.dirname(video_path[0])

print(f"--- Initializing SuperAnimal on 16GB GPU ---")

# 2. Run Inference with Maxed Hyperparameters
deeplabcut.video_inference_superanimal(
    video_path, 
    superanimal_name="superanimal_topviewmouse",
    model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2",
    videotype=".mp4", 
    video_adapt=False,       # Skip adaptation to save hours and disk space
    # GPU only at 20-30% utilization for this
    batch_size = 256,           # High batch size for HRNet pose estimation
    # Took 21 minutes with detector_batch_size = 4
    #
    # CPU is the Bottlneck
    detector_batch_size=2,  # Speed up the mouse detection phase
    create_labeled_video=True,# Generates the .mp4 with dots
    max_individuals=1
)

print(f"--- Finalizing Files ---")

# 3. Generate the CSV for VAME
# DLC 3.0 sometimes saves .h5 by default; this ensures you have the CSV
deeplabcut.analyze_videos_converth5_to_csv(video_dir, videotype=".mp4")

print(f"--- Process Complete ---")
print(f"Check your video folder for the labeled video and CSV.")