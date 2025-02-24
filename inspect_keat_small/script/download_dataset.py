import os
import urllib.request
import zipfile
import shutil

from pathlib import Path


def download_and_extract_mrl(target_dir=None):
    """
    Downloads and extracts the MRL-2021 dataset

    Args:
        target_dir (str or Path, optional): Directory where dataset should be extracted.
            Will be created if it doesn't exist.
            If None, uses the script's directory.

    Returns:
        bool: True if successful, False otherwise
        Path: Path to the extracted dataset directory (target_dir/MRL-2021)
    """
    try:
        # Set up paths
        if target_dir is None:
            target_dir = Path(__file__).parent
        else:
            target_dir = Path(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)

        zip_path = target_dir / "mrl_2021.zip"
        temp_extract_path = target_dir / "temp_extract"
        final_path = target_dir / "MRL-2021"

        # Download the zip file if it doesn't exist
        if not zip_path.exists():
            url = "https://github.com/emorynlp/MRL-2021/archive/refs/heads/master.zip"
            print(f"Downloading MRL-2021 dataset from {url}...")
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete!")

        # Create temp directory for extraction
        temp_extract_path.mkdir(exist_ok=True)

        # Extract to temp directory
        print(f"Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_path)

        # Move contents from MRL-2021-master to final location
        master_path = temp_extract_path / "MRL-2021-master"
        if master_path.exists():
            if final_path.exists():
                shutil.rmtree(final_path)
            shutil.move(master_path, final_path)
            print("Moved files to correct location")

        # Cleanup
        if zip_path.exists():
            os.remove(zip_path)
        if temp_extract_path.exists():
            shutil.rmtree(temp_extract_path)
        print("Cleaned up temporary files")

        print(f"MRL-2021 dataset is ready to use at {final_path}!")
        return True, final_path

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return False, None


if __name__ == "__main__":
    # Example usage with specific target directory
    target_dir = Path(__file__).parent.parent.parent / "lib" / "data"
    success, dataset_path = download_and_extract_mrl(target_dir)
    if success:
        print(f"Dataset downloaded to: {dataset_path}")
    else:
        print("Failed to download dataset")
