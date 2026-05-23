#!/bin/bash
# Upload VAME project to cluster
# Usage: bash upload_to_cluster.sh

CLUSTER="cluster"
REMOTE_DIR="~/kaplan_research/VAME"
LOCAL_DIR="/mnt/c/Users/zuria/Kaplan/VAME"

echo "=============================="
echo " VAME Cluster Upload"
echo "=============================="
echo "Destination: $CLUSTER:$REMOTE_DIR"
echo ""

# Step 1 — Create remote folders and clone code from GitHub
echo "[1/6] Setting up remote directories and cloning repo..."
ssh $CLUSTER "
  mkdir -p $REMOTE_DIR && \
  cd $REMOTE_DIR && \
  git clone https://github.com/Zuriahn-Yun/kaplan-mouse-behavior-vame . 2>/dev/null || git pull && \
  mkdir -p $REMOTE_DIR/Open-Field-Test/data/raw \
            $REMOTE_DIR/Open-Field-Test/data/processed \
            $REMOTE_DIR/Open-Field-Test/data/train \
            $REMOTE_DIR/Open-Field-Test/model \
            $REMOTE_DIR/Open-Field-Test/results \
            $REMOTE_DIR/Open-Field-Test/logs
"
echo "Done."
echo ""

# Step 2 — Raw data + videos (15 GB)
echo "[2/6] Uploading raw data and videos (~15 GB)..."
rsync -avz --progress \
  "$LOCAL_DIR/Open-Field-Test/data/raw/" \
  "$CLUSTER:$REMOTE_DIR/Open-Field-Test/data/raw/"
echo "Done."
echo ""

# Step 3 — Processed .nc files
echo "[3/6] Uploading processed egocentric files..."
rsync -avz --progress \
  "$LOCAL_DIR/Open-Field-Test/data/processed/" \
  "$CLUSTER:$REMOTE_DIR/Open-Field-Test/data/processed/"
echo "Done."
echo ""

# Step 4 — Training arrays
echo "[4/6] Uploading training arrays..."
rsync -avz --progress \
  "$LOCAL_DIR/Open-Field-Test/data/train/" \
  "$CLUSTER:$REMOTE_DIR/Open-Field-Test/data/train/"
echo "Done."
echo ""

# Step 5 — Model weights and snapshots
echo "[5/6] Uploading model weights..."
rsync -avz --progress \
  "$LOCAL_DIR/Open-Field-Test/model/" \
  "$CLUSTER:$REMOTE_DIR/Open-Field-Test/model/"
echo "Done."
echo ""

# Step 6 — Results
echo "[6/6] Uploading segmentation results (~20 GB)..."
rsync -avz --progress \
  "$LOCAL_DIR/Open-Field-Test/results/" \
  "$CLUSTER:$REMOTE_DIR/Open-Field-Test/results/"
echo "Done."
echo ""

echo "=============================="
echo " Upload complete!"
echo " Next: ssh cluster, then run:"
echo "   cd $REMOTE_DIR"
echo "   conda env create -f environment.yml"
echo "=============================="
