# Transformer-based Models for Unsupervised Anomaly Segmentation in Brain MR Images

Official implementation of [Transformer-based Models for Unsupervised Anomaly Segmentation in Brain MR Images]([https://arxiv.org/pdf/2207.02059.pdf](https://link.springer.com/chapter/10.1007/978-3-031-33842-7_3#citeas)).

[Paper accepted in the International MICCAI Brainlesion 2022 Workshop]([https://link.springer.com/conference/iwb](https://link.springer.com/book/10.1007/978-3-031-33842-7))

```
@InProceedings{10.1007/978-3-031-33842-7_3,
    author="Ghorbel, Ahmed
    and Aldahdooh, Ahmed
    and Albarqouni, Shadi
    and Hamidouche, Wassim",
    editor="Bakas, Spyridon
    and Crimi, Alessandro
    and Baid, Ujjwal
    and Malec, Sylwia
    and Pytlarz, Monika
    and Baheti, Bhakti
    and Zenk, Maximilian
    and Dorent, Reuben",
    title="Transformer Based Models for Unsupervised Anomaly Segmentation in Brain MR Images",
    booktitle="Brainlesion:  Glioma, Multiple Sclerosis, Stroke  and Traumatic Brain Injuries",
    year="2023",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="25--44",
    isbn="978-3-031-33842-7"
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

All packages used in this repository are listed in [requirements.txt](https://github.com/ahmedgh970/brain-anomaly-seg/requirements.txt).
To install those, run:
```
pip3 install -r requirements.txt
```


## Folder Structure
  ```
  brain-anomaly-seg/
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
  └── scripts/ - small utility scripts
      ├── utils.py
      └── ...    
  ```

## Usage
### CLI Usage
Every model can be trained and tested individually using the scripts which are provided in the `models/*` folders.


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
