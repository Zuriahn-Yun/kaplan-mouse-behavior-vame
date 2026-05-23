import os
import deeplabcut
import sys
# 1. Define your list of video paths
video_list = [
    r"Subject 2 - female GO Vehicle OFT.mp4",
]
missing_videos = [v for v in video_list if not os.path.exists(v)]

if missing_videos:
    print("ERROR: The following video files were not found")
    for vid in missing_videos:
        print(f"  - Missing: {vid}")
    print("\nPlease check your file paths and try again. Script exiting.")
    sys.exit() # Stops the script before it starts processing
else:
    print("All video paths verified. Starting batch process...")

# region of interest, can be manually gotten for a set of videos with DLC Cooridnates.py
my_roi = [825, 1327, 246, 747]
failed_videos = []

print(f"--- Initializing SuperAnimal Batch Processing ---")

for video_path in video_list:
    # Get the directory for the CSV conversion later
    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    
    print(f"\nProcessing: {video_name}...")
    
    try:
        # 2. Run Inference
        # Note: we pass [video_path] as a list containing one string
        deeplabcut.video_inference_superanimal(
            [video_path], 
            superanimal_name="superanimal_topviewmouse",
            model_name="hrnet_w32",
            detector_name="fasterrcnn_resnet50_fpn_v2",
            videotype=".mp4", 
            video_adapt=False,
            batch_size=256,
            detector_batch_size=2,
            create_labeled_video=True,
            max_individuals=1,
            cropping=my_roi,
        )
        
        print(f"Successfully processed: {video_name}")

    except Exception as e:
        print(f"!!! FAILED: {video_name} !!!")
        print(f"Error details: {e}")
        failed_videos.append(video_name)

# --- Final Summary ---
print(f"\n--- Batch Process Complete ---")
if failed_videos:
    print(f"The following videos failed:")
    for vid in failed_videos:
        print(f"- {vid}")
else:
    print("All videos processed successfully!")