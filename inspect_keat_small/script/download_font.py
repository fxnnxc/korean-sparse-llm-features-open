# Standard library imports
import os
import urllib.request
import platform
import subprocess
import shutil
import traceback
from pathlib import Path

# Third party imports
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def download_and_install_noto_font(save_test_image=True):
    """
    Downloads and installs Noto Sans CJK KR font for matplotlib

    Args:
        save_test_image (bool): Whether to save test image to file

    Returns:
        tuple: (success (bool), figure (matplotlib.figure.Figure or None))
    """

    # Font URLs - using Google Fonts API
    font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Korean/NotoSansCJKkr-Regular.otf"
    temp_dir = Path(__file__).parent / "temp_fonts"
    temp_dir.mkdir(parents=True, exist_ok=True)
    font_path = temp_dir / "NotoSansCJKkr-Regular.otf"

    try:
        # Download font if it doesn't exist
        if not font_path.exists():
            print("Downloading Noto Sans CJK KR font...")
            urllib.request.urlretrieve(font_url, font_path)
            print("Font download complete!")

        # Get system font directory
        if platform.system() == 'Windows':
            font_dir = Path(os.environ['WINDIR']) / 'Fonts'
        elif platform.system() == 'Darwin':  # macOS
            font_dir = Path.home() / 'Library' / 'Fonts'
        else:  # Linux
            font_dir = Path.home() / '.local' / 'share' / 'fonts'
            font_dir.mkdir(parents=True, exist_ok=True)

        # Install font to system directory
        system_font_path = font_dir / font_path.name
        if not system_font_path.exists():
            if platform.system() == 'Windows':
                shutil.copy2(font_path, system_font_path)
            else:
                if system_font_path.exists():
                    os.remove(system_font_path)
                shutil.copy2(font_path, system_font_path)

        # Update font cache on Linux
        if platform.system() == 'Linux':
            try:
                subprocess.run(['fc-cache', '-f', '-v'], check=True)
            except subprocess.CalledProcessError:
                print("Warning: Failed to update font cache. Font might not be immediately available.")

        # Clear and reload matplotlib font cache
        fm.fontManager.addfont(str(system_font_path))

        # Force matplotlib to use the font
        font_names = [f.name for f in fm.fontManager.ttflist]
        print("Available fonts:", [f for f in font_names if 'Noto' in f])

        # Try different possible font names
        font_options = ['Noto Sans CJK KR', 'Noto Sans CJK KR Regular', 'NotoSansCJKkr-Regular']
        selected_font = None
        for font in font_options:
            if font in font_names:
                selected_font = font
                break

        if selected_font is None:
            print("Warning: Could not find Noto font in system. Using default font.")
            selected_font = 'DejaVu Sans'

        # Configure matplotlib
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']

        # Test the font
        print("Testing font installation...")
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, '한글 테스트 Text', fontsize=20, ha='center', va='center')
        ax.set_axis_off()

        if save_test_image:
            # Save with higher DPI for better quality
            test_image_path = Path(__file__).parent / 'font_test.png'
            plt.savefig(test_image_path, dpi=300, bbox_inches='tight')
            print(f"Font test image saved to {test_image_path}")

        # Cleanup temporary files
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary files")

        return True, fig

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        traceback.print_exc()
        # Cleanup on error
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return False, None


if __name__ == "__main__":
    success, fig = download_and_install_noto_font()
    if success:
        plt.close(fig)  # Close the figure when running as script
