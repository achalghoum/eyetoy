#!/bin/bash
#
# Script to download ImageNet dataset tarballs.
# Will download the required tarballs into the 'imagenet' directory.
# Based on documentation requiring:
# - ILSVRC2012_devkit_t12.tar.gz
# - ILSVRC2012_img_train.tar
# - ILSVRC2012_img_val.tar

set -e  # Exit on error

# Function to handle cleanup on exit and errors
cleanup_downloads() {
    echo "Cleaning up any temporary files..."
    find . -name "*.wget-tmp" -delete
}

# Set up trap for various signals
trap cleanup_downloads EXIT INT TERM

# Function to download with axel with retries
download_with_axel() {
    local url=$1
    local output_file=$2
    local max_retries=100
    local retry_count=0
    local success=false
    # Fixed number of connections
    local connections=256
    
    echo "Downloading $output_file with axel..."
    
    # Check if file exists and has non-zero size
    if [ -f "$output_file" ] && [ -s "$output_file" ]; then
        echo "File $output_file already exists with non-zero size, skipping download."
        return 0
    fi
    
    # Check for partial downloads (.st extension - axel state file)
    if [ -f "${output_file}.st" ]; then
        echo "Found incomplete download state file ${output_file}.st, will resume download."
    fi
    
    # Try downloading
    while [ $retry_count -lt $max_retries ] && [ "$success" = false ]; do
        echo "Attempt $(($retry_count + 1))..."
        
        if axel -n $connections -a -k --timeout=-1 --insecure -o "$output_file" "$url"; then
            success=true
            echo "Download of $output_file completed successfully."
            break
        else
            retry_count=$((retry_count + 1))
            echo "Download attempt $retry_count failed."
            
            # Try curl as an alternative on every 10th retry
            if [ $((retry_count % 10)) -eq 0 ]; then
                echo "Trying curl as alternative..."
                if curl -C - --retry 5 --retry-delay 10 --retry-max-time 0 -k -L -o "$output_file" "$url"; then
                    success=true
                    echo "Download completed successfully with curl."
                    break
                fi
            fi
            
            # Try wget as final fallback on last attempt
            if [ $retry_count -eq $max_retries ]; then
                echo "Trying wget as final fallback..."
                if wget -c --retry-connrefused --waitretry=1 --read-timeout=120 --timeout=-1 -t 5 "$url" -O "$output_file" --no-check-certificate; then
                    success=true
                    echo "Download completed successfully with wget."
                    break
                fi
            fi
            
            # Wait before retrying
            sleep_time=$((5 + retry_count / 2))
            if [ $sleep_time -gt 30 ]; then
                sleep_time=30
            fi
            echo "Waiting for $sleep_time seconds before retrying..."
            sleep $sleep_time
        fi
    done
    
    if [ "$success" = false ]; then
        echo "Failed to download $output_file after $max_retries attempts."
        return 1
    fi
    
    return 0
}

# Check if axel is installed and install if not exists
if ! command -v axel &> /dev/null; then
    echo "axel could not be found. Attempting to install automatically..."
    
    # Detect OS and install accordingly
    if [ -f /etc/debian_version ]; then
        # Debian/Ubuntu
        echo "Detected Debian/Ubuntu, installing with apt..."
        sudo apt-get update && sudo apt-get install -y axel
    elif [ -f /etc/redhat-release ]; then
        # RHEL/CentOS/Fedora
        echo "Detected RHEL/CentOS/Fedora, installing with yum..."
        sudo yum install -y axel
    elif command -v brew &> /dev/null; then
        # macOS with Homebrew
        echo "Detected macOS with Homebrew, installing with brew..."
        brew install axel
    elif command -v pacman &> /dev/null; then
        # Arch Linux
        echo "Detected Arch Linux, installing with pacman..."
        sudo pacman -S --noconfirm axel
    elif command -v dnf &> /dev/null; then
        # Fedora with DNF
        echo "Detected Fedora with DNF, installing with dnf..."
        sudo dnf install -y axel
    elif command -v zypper &> /dev/null; then
        # openSUSE
        echo "Detected openSUSE, installing with zypper..."
        sudo zypper install -y axel
    else
        echo "Could not automatically install axel. Please install manually:"
        echo "For Ubuntu/Debian: sudo apt-get install axel"
        echo "For CentOS/RHEL: sudo yum install axel"
        echo "For macOS: brew install axel"
        echo "For Windows: Use Windows Subsystem for Linux or install manually"
        exit 1
    fi
fi

# Setup main directory
IMAGENET_DIR="imagenet"
mkdir -p $IMAGENET_DIR
cd $IMAGENET_DIR

echo "==> Downloading ImageNet dataset tarballs..."

## Download the data

# Devkit
echo "Downloading Devkit..."
download_with_axel "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz" "ILSVRC2012_devkit_t12.tar.gz"

# Validation set
echo "Downloading validation set..."
download_with_axel "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar" "ILSVRC2012_img_val.tar"

# Training set
echo "Downloading training set..."
download_with_axel "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar" "ILSVRC2012_img_train.tar"

# Go back to the original directory
cd ..

echo "==> ImageNet tarball download process finished."
echo "Files are located in the '$IMAGENET_DIR/' directory."
