<p align="center">
  <a href="https://github.com/SeatizenDOI/DinoVdeau/graphs/contributors"><img src="https://img.shields.io/github/contributors/SeatizenDOI/DinoVdeau" alt="GitHub contributors"></a>
  <a href="https://github.com/SeatizenDOI/DinoVdeau/network/members"><img src="https://img.shields.io/github/forks/SeatizenDOI/DinoVdeau" alt="GitHub forks"></a>
  <a href="https://github.com/SeatizenDOI/DinoVdeau/issues"><img src="https://img.shields.io/github/issues/SeatizenDOI/DinoVdeau" alt="GitHub issues"></a>
  <a href="https://github.com/SeatizenDOI/DinoVdeau/blob/master/LICENSE"><img src="https://img.shields.io/github/license/SeatizenDOI/DinoVdeau" alt="Licenses"></a>
  <a href="https://github.com/SeatizenDOI/DinoVdeau/pulls"><img src="https://img.shields.io/github/issues-pr/SeatizenDOI/DinoVdeau" alt="GitHub pull requests"></a>
  <a href="https://github.com/SeatizenDOI/DinoVdeau/stargazers"><img src="https://img.shields.io/github/stars/SeatizenDOI/DinoVdeau" alt="GitHub stars"></a>
  <a href="https://github.com/SeatizenDOI/DinoVdeau/watchers"><img src="https://img.shields.io/github/watchers/SeatizenDOI/DinoVdeau" alt="GitHub watchers"></a>
</p>

<div align="center">
  <img src="images/DinoVd_eau_architecture.png" alt="Project logo" width="700">
  <p align="center">A classification framework to enhance underwater computer vision models.</p>
  <a href="https://github.com/SeatizenDOI/DinoVdeau">View framework</a>
  ·
  <a href="https://github.com/SeatizenDOI/DinoVdeau/issues">Report Bug</a>
  ·
  <a href="https://github.com/SeatizenDOI/DinoVdeau/issues">Request Feature</a>
  <h1></h1>
</div>

# DinoVd'eau: Underwater Multilabel Image Classification

This repository contains all necessary components for training and evaluating DinoVd'eau model, a deep learning model fine-tuned for underwater multilabel image classification. It leverages the dinov2 architecture and is customized for high precision in identifying diverse marine species.

A demo of the model can be found <a href="https://huggingface.co/spaces/lombardata/Victor_DinoVdEau_Image_Classification">here</a>.

It is a slightly adapted version of the original [DINOv2](https://arxiv.org/abs/2304.07193), GitHub [repository](https://github.com/facebookresearch/dinov2/).


## Major Frameworks and Libraries

This section lists the key frameworks and libraries used to create the models included in the project:

* [![PyTorch](https://img.shields.io/badge/PyTorch-%23ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/) - A deep learning library used for constructing and training neural network models.
* [![scikit-learn](https://img.shields.io/badge/scikit--learn-%23f89a36.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org) - Utilized for various machine learning tools for data mining and data analysis.
* [![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD43B?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/) - Provides thousands of pre-trained models to perform tasks on different modalities such as text, vision, and audio.

## Installation

To ensure a consistent environment for all users, this project uses a Conda environment defined in a `requirements.yml` file. Follow these steps to set up your environment:

1: **Clone the Repository** : First, clone the `DinoVd'eau` repository
```bash
git clone https://github.com/SeatizenDOI/DinoVdeau.git
cd DinoVdeau
```

2: **Create and Activate Conda Environment** : Next, create a Conda environment using the `requirements.yml` file and activate it
```bash
conda env create -f requirements.yml
conda activate dinovdeau_env
```

3: **Install PyTorch** : Finally, install PyTorch. It is recommended to install PyTorch with CUDA support for optimal performance. Follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch with the appropriate options for your system.

Here is an example command to install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4: **Create configiguration file** : At root folder, you need to create a config file called `config.json` with this parameters :
```json
{
  "ANNOTATION_PATH": "Path to annotation folder with you train.csv, val.csv, test.csv files",
  "IMG_PATH": "Path to images folder",
  "MODEL_PATH": "Path to store the new model",
  "MODEL_NAME": "Path to model when you want to resume",
  "HUGGINGFACE_TOKEN": "YOUR API TOKEN",
  "LOCAL_MODEL_PATH": "/mnt/disk_victorlebos/data/datarmor/models/local_models/dinov2-large/"
}
```


By following these steps, you will set up the necessary environment to work with `DinoVd'eau`.

## Usage

To start the training process, navigate to the project root and execute:

```bash
python main.py [OPTIONS]
```

Where `[OPTIONS]` can include:

### Training Parameters

- `--image_size`: Image size for both dimensions. (default: 518)
- `--batch_size`: Batch size for training and evaluation. (default: 32)
- `--epochs`: Number of training epochs. (default: 150)
- `--initial_learning_rate`: Initial learning rate for the optimizer. (default: 0.001)
- `--weight_decay`: Weight decay factor for the optimizer. (default: 0.0001)
- `--early_stopping_patience`: Number of epochs with no improvement after which training will be stopped. (default: 10)
- `--patience_lr_scheduler`: Number of epochs to wait before reducing the learning rate. (default: 5)
- `--factor_lr_scheduler`: Factor by which the learning rate will be reduced. (default: 0.1)

### Model Configuration

- `--model_name`: Name or path of the pretrained model from Hugging Face. (default: "facebook/dinov2-large")
- `--new_model_name`: Name to assign to the saved model. (default: "test_dino")
- `--training_type`: Training strategy, choose between "multilabel" or "monolabel". (default: "multilabel")
- `--no_custom_head`: If set, uses a linear classification head instead of a custom one.

### Training Options

- `--no_freeze`: If set, the model backbone will not be frozen (i.e., it will be fine-tuned).
- `--no_data_aug`: If set, disables data augmentation during training.
- `--test_data`: If set, uses a small subset of data to test the workflow.
- `--resume`: If set, resumes training from the last checkpoint.

### Global Options

- `--disable_web`: If set, disables any connection to the web.
- `--config_path`: Path to the configuration file. (default: "config.json")

## Team

DinoVd'eau is a community-driven project with several skillful people contributing to it.  
DinoVd'eau is currently maintained by [Matteo Contini](https://github.com/lombardata) with major contributions coming from [Alexis Joly](https://orcid.org/0000-0002-2161-9940), [Sylvain Bonhommeau](https://orcid.org/0000-0002-0882-5918), [Victor Illien](https://github.com/Gouderg), [César Leblanc](https://orcid.org/0000-0002-5682-8179), and the amazing people from the [Ifremer DOI Team](https://ocean-indien.ifremer.fr/) in various forms and means.

## Contributing

Contributions are welcome! To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Commit your changes with clear, descriptive messages.
4. Push your branch and submit a pull request.

## License

This framework is distributed under the CC0-1.0 license. See `LICENSE.txt` for more information.

## Citing DinoVd'eau

If you find this repository useful, please consider giving a star :star: and citation :fish::

```
@article{Contini2025,
   author = {Matteo Contini and Victor Illien and Mohan Julien and Mervyn Ravitchandirane and Victor Russias and Arthur Lazennec and Thomas Chevrier and Cam Ly Rintz and Léanne Carpentier and Pierre Gogendeau and César Leblanc and Serge Bernard and Alexandre Boyer and Justine Talpaert Daudon and Sylvain Poulain and Julien Barde and Alexis Joly and Sylvain Bonhommeau},
   doi = {10.1038/s41597-024-04267-z},
   issn = {2052-4463},
   issue = {1},
   journal = {Scientific Data},
   pages = {67},
   title = {Seatizen Atlas: a collaborative dataset of underwater and aerial marine imagery},
   volume = {12},
   url = {https://doi.org/10.1038/s41597-024-04267-z},
   year = {2025},
}
```


```
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```
<div align="center">
  <img src="https://github.com/SeatizenDOI/.github/blob/main/images/logo_partenaire_2.png?raw=True" alt="Partenaire logo" width="700">
</div>
