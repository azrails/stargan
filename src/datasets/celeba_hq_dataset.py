import zipfile
import gdown
import re
import requests
import logging
from pathlib import Path
from tqdm.auto import tqdm
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH
from torchvision.io import read_image
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)

class CelebaHQDataset(BaseDataset):
    """
    Custom Dataset class for the CelebA dataset. Automatically downloads
    data and annotations from Google Drive if not found in the specified root_dir.
    """
    download_url = "https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=1"
    def __init__(
            self,
            data_dir: str | Path | None = None,
            *args,
            **kwargs
            ):
        if data_dir is None:
            data_dir = ROOT_PATH / "datasets" / "celeba"
            data_dir.mkdir(exist_ok=True, parents=True)
        
        data_dir = Path(data_dir)
        self._data_dir = data_dir

        index = self._get_index()

        super().__init__(index, *args, **kwargs)

    def load_object(self, path):
        return read_image(path)

    def _get_index(self):
        #downloading images
        image_folder = self._data_dir / 'celeba_hq'
        zip_file = self._data_dir / 'celeba_hq.zip'
        if not image_folder.exists():
            if not zip_file.exists():
                logger.debug("Downloading dataset data...")
                response = requests.get(self.download_url, stream=True)
                with open('celeba_hq.zip', 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.debug(f"Extracting {zip_file}...")
            with zipfile.ZipFile(zip_file, "r") as z:
                z.extractall(self._data_dir)
        
        
