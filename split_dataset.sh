#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
URL="https://storage.lczero.org/files/training_data/test80/training-run1-test80-20240401-0017.tar"
TAR_FILE="dataset.tar"
TEMP_DIR="tmp_extracted_dataset"
TRAIN_DIR="training-dataset/train"
VAL_DIR="training-dataset/val"

# Number of files for training (N) and validation (M)
# You can modify these values or pass them as arguments ($1 and $2)
N=${1:-10000}
M=${2:-2000}

echo "==> Creating directories..."
mkdir -p "$TRAIN_DIR" "$VAL_DIR" "$TEMP_DIR"

echo "==> Downloading dataset from $URL..."
if [ ! -f "$TAR_FILE" ]; then
    wget -O "$TAR_FILE" "$URL"
else
    echo "Found existing $TAR_FILE, skipping download."
fi

echo "==> Extracting tar file into temporary directory..."
# Using tar to extract into TEMP_DIR
tar -xf "$TAR_FILE" -C "$TEMP_DIR"

echo "==> Locating .gz files..."
# Find all .gz files recursively inside the extracted folder
mapfile -t GZ_FILES < <(find "$TEMP_DIR" -type f -name "*.gz")

TOTAL_FOUND=${#GZ_FILES[@]}
echo "Found $TOTAL_FOUND .gz files in the archive."

# Validate if we have enough files
REQUIRED=$((N + M))
if [ "$TOTAL_FOUND" -lt "$REQUIRED" ]; then
    echo "Error: Requested N ($N) + M ($M) = $REQUIRED files, but only found $TOTAL_FOUND files in the archive."
    exit 1
fi

echo "==> Shuffling files randomly..."
# Shuffle the file list to ensure random train/val distribution
SHUFFLED_FILES=($(printf "%s\n" "${GZ_FILES[@]}" | shuf))

echo "==> Moving $N files to $TRAIN_DIR..."
for ((i=0; i<N; i++)); do
    mv "${SHUFFLED_FILES[$i]}" "$TRAIN_DIR/"
done

echo "==> Moving $M files to $VAL_DIR..."
for ((i=N; i<N+M; i++)); do
    mv "${SHUFFLED_FILES[$i]}" "$VAL_DIR/"
done

echo "==> Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
# Optional: rm -f "$TAR_FILE"  # Uncomment if you want to delete the downloaded tar archive

echo "Done! Successfully placed $N files in $TRAIN_DIR and $M files in $VAL_DIR."
