# Transformer-based Models for Unsupervised Anomaly Segmentation in Brain MR Images

You can find more details in our preprint paper: [Transformer-based Models for Unsupervised Anomaly Segmentation in Brain MR Images](https://arxiv.org/pdf/2207.02059.pdf). 
```
@article{ghorbel2022transformer,
  title={Transformer based Models for Unsupervised Anomaly Segmentation in Brain MR Images},
  author={Ghorbel, Ahmed and Aldahdooh, Ahmed and Albarqouni, Shadi and Hamidouche, Wassim},
  journal={arXiv preprint arXiv:2207.02059},
  year={2022}
}
```


* [Transformers_Unsupervised_Anomaly_Segmentation](#Transformers_Unsupervised_Anomaly_Segmentation)
  * [Requirements](#requirements)
  * [Folder Structure](#folder-structure)
  * [Usage](#usage)
      * [CLI-Usage](#cli-usage)
      * [Google Colab Usage](#google-colab-usage)
  * [Disclaimer](#disclaimer)
  * [Reference](#reference)
  * [License](#license)
    
<!-- /code_chunk_output -->


## Tags
<code>MICCAI BrainLes 2022 Workshop</code>, <code>Transformer</code>, <code>Autoencoder</code>, <code>TensorFlow</code>, <code>Keras</code>, <code>Anomaly Segmentation</code>, <code>Unsupervised</code>, <code>Neuroimaging</code>, <code>Deeplearning</code>


## Requirements
* <code>Python >= 3.6</code>

All packages used in this repository are listed in [requirements.txt](https://github.com/ahmedgh970/Transformers_Unsupervised_Anomaly_Segmentation/requirements.txt).
To install those, run:
```
pip3 install -r requirements.txt
```


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
  ├── saved/  - saving folder
  │
  └── scripts/ - small utility scripts
      ├── utils.py
      └── ...    
  ```

## Usage
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
