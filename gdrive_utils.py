"""
Google Drive utilities for downloading datasets and models.
"""

import streamlit as st
import gdown
import os
import zipfile
import shutil


def download_from_gdrive(gdrive_id: str, output_path: str, is_folder: bool = True):
    """
    Download file or folder from Google Drive.

    Args:
        gdrive_id: Google Drive file/folder ID
        output_path: Local path to save
        is_folder: True for folder, False for single file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if is_folder:
            # Download folder
            st.info(f"📥 Downloading folder from Google Drive...")
            gdown.download_folder(
                id=gdrive_id,
                output=output_path,
                quiet=False,
                use_cookies=False
            )
        else:
            # Download single file
            st.info(f"📥 Downloading file from Google Drive...")
            url = f"https://drive.google.com/uc?id={gdrive_id}"
            gdown.download(url, output_path, quiet=False)

        st.success(f"✅ Download complete: {output_path}")
        return True

    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        st.error("Make sure the Google Drive link is shared publicly (Anyone with the link)")
        return False


def download_and_extract_zip(gdrive_id: str, output_folder: str):
    """
    Download a zip file from Google Drive and extract it.

    Args:
        gdrive_id: Google Drive file ID (zip file)
        output_folder: Folder to extract contents

    Returns:
        bool: True if successful
    """
    try:
        zip_path = "temp_download.zip"

        # Download zip
        st.info("📥 Downloading dataset (zip)...")
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, zip_path, quiet=False)

        # Extract
        st.info("📦 Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)

        # Cleanup
        os.remove(zip_path)

        st.success(f"✅ Dataset extracted to: {output_folder}")
        return True

    except Exception as e:
        st.error(f"❌ Extraction failed: {str(e)}")
        return False


def ensure_dataset_exists(dataset_dir: str, gdrive_id: str, force_download: bool = False):
    """
    Ensure dataset exists locally, download if needed.

    Args:
        dataset_dir: Local dataset directory
        gdrive_id: Google Drive folder ID
        force_download: Force re-download even if exists

    Returns:
        bool: True if dataset is available
    """
    # Check if already exists
    if os.path.exists(dataset_dir) and not force_download:
        if os.listdir(dataset_dir):  # Not empty
            st.sidebar.success(f"✅ Dataset found locally: {dataset_dir}")
            return True

    # Need to download
    st.sidebar.warning(f"⚠️ Dataset not found locally")

    if not gdrive_id or gdrive_id == "YOUR_FOLDER_ID_HERE":
        st.sidebar.error("❌ Google Drive ID not configured in config.py")
        st.sidebar.info("Please update DATASET_GDRIVE_ID in config.py")
        return False

    # Download from Google Drive
    with st.spinner("Downloading dataset from Google Drive..."):
        success = download_from_gdrive(gdrive_id, dataset_dir, is_folder=True)

    return success


def ensure_model_exists(model_path: str, gdrive_id: str = None):
    """
    Ensure model file exists locally, download if needed.

    Args:
        model_path: Local model file path
        gdrive_id: Google Drive file ID (optional)

    Returns:
        bool: True if model is available
    """
    # Check if already exists
    if os.path.exists(model_path):
        st.sidebar.success(f"✅ Model found: {model_path}")
        return True

    # Need to download
    if not gdrive_id or gdrive_id == "YOUR_MODEL_FILE_ID_HERE":
        st.sidebar.warning(f"⚠️ Model not found: {model_path}")
        st.sidebar.info("Please upload your model file or configure MODEL_GDRIVE_ID")
        return False

    # Download from Google Drive
    st.sidebar.info("📥 Downloading model from Google Drive...")
    with st.spinner("Downloading model..."):
        success = download_from_gdrive(gdrive_id, model_path, is_folder=False)

    return success