# Transformers for Unsupervised Anomaly Segmentation in Neuroimaging

This repository contains the code for our paper on [Transformers for Unsupervised Anomaly Segmentation in Neuroimaging](https://www.sciencedirect.com). 
If you use any of our code, please cite:
```
@article{ahghorbe2021,
  title = {Transformers for Unsupervised Anomaly Segmentation in Neuroimaging},
  author = {Ghorbel, Ahmed and Hamidouche, Wassim and Aldahdooh, Ahmed and Albarqouni, Shadi},
  url = {http://arxiv.org},
  year = {2022}
}
```

* [Transformers for Unsupervised Anomaly Segmentation in Neuroimaging](#transformers-for-unsupervised-anomaly-segmentation-in-neuroimaging)
  * [Requirements](#requirements)
  * [Folder Structure](#folder-structure)
  * [Usage](#usage)
      * [CLI-Usage](#cli-usage)
      * [Google Colab Usage](#google-colab-usage)
  * [Disclaimer](#disclaimer)
  * [Reference](#reference)
  * [License](#license)
    
<!-- /code_chunk_output -->


## Requirements
* Python >= 3.6

All packages used in this repository are listed in [requirements.txt](https://github.com/ahmedgh970/Transformers_Unsupervised_Anomaly_Segmentation/requirements.txt).
To install those, run `pip3 install -r requirements.txt`


## Folder Structure
  ```
  Transformers_Unsupervised_Anomaly_Segmentation/
  │
  ├── Transformers_Unsupervised_Anomaly_Segmentation.ipynb - Jupyter notebook to work on Google Colab
  │
  ├── data/
  │   └── data.txt  - datasets descriptions and download link
  │
  ├── models/ - Models defining, training and evaluating
  │   ├── Autoencoders/
  │       ├── DCAE.py
  │       └── ...
  │   ├── Latent Variable models/
  │       ├── VAE.py
  │       └── ...
  │   └── Transformer based models/
  │       ├── B_TAE.py
  │       └── ...
  │  
  └── scripts/ - small utility scripts
      ├── utils.py
      └── ...    
  ```


## Usage
All the preprocessed datasets that we utilized are available on this drive link: https://drive.google.com/file/d/11Bj7ATQtxLt7PyL3fqyyeXqNNrRqgS9K/view?usp=sharing

### CLI Usage
Every model can be trained and tested individually using the scripts which are provided in the `models/*` folders.

### Google Colab Usage
Training can be started by importing `Transformers_Unsupervised_Anomaly_Segmentation.ipynb` in [Google Colab](http://colab.research.google.com).
This github repository is linked and can directly loaded into the notebook. However, the datasets have to be stored so that Google Colab can access them. 
Either uploading by a zip-file or uploading it to Google Drive and mounting the drive.


## Disclaimer
Please do not hesitate to open an issue to inform of any problem you may find within this repository.


## Reference
This project is inspired by the comparative study paper on [Autoencoders for Unsupervised Anomaly Segmentation in Brain MR Images: A Comparative Study](https://www.sciencedirect.com/science/article/abs/pii/S1361841520303169).

```
@article{baur2021autoencoders,
  title={Autoencoders for unsupervised anomaly segmentation in brain mr images: A comparative study},
  author={Baur, Christoph and Denner, Stefan and Wiestler, Benedikt and Navab, Nassir and Albarqouni, Shadi},
  journal={Medical Image Analysis},
  pages={101952},
  year={2021},
  publisher={Elsevier}
}
```


## License
This project is licensed under the GNU General Public License v3.0. See LICENSE for more details
