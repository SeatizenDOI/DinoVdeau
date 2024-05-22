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


## Project Structure

This repository is organized as follows to facilitate model training and evaluation:

```
.
├── config
│   ├── arguments.py           # Defines command-line arguments for training
├── config.json                # General configuration for training parameters
├── data
│   ├── data_loading.py        # Handles data loading
│   ├── data_preprocessing.py  # Preprocesses data for model input
├── main.py                    # Main script to start training and evaluation
├── model
│   ├── model_setup.py         # Sets up the model architecture and configuration
├── README.md                  # Project documentation and instructions
├── requirements.yml           # Conda environment file to reproduce the project environment
└── utils
    ├── evaluation.py          # Evaluation utilities for model performance
    ├── training.py            # Utilities to facilitate the training process
    └── utils.py               # General utilities for model card.
```

## Major Frameworks and Libraries

This section lists the key frameworks and libraries used to create the models included in the project:

* [![PyTorch](https://img.shields.io/badge/PyTorch-%23ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/) - A deep learning library used for constructing and training neural network models.
* [![scikit-learn](https://img.shields.io/badge/scikit--learn-%23f89a36.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org) - Utilized for various machine learning tools for data mining and data analysis.
* [![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD43B?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/) - Provides thousands of pre-trained models to perform tasks on different modalities such as text, vision, and audio.

## Installation

To ensure a consistent environment for all users, this project uses a Conda environment defined in a `requirements.yml` file. Follow these steps to set up your environment:

1. **Install Conda:** If you do not have Conda installed, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

2. **Create the Conda Environment:** Navigate to the root of the project directory and run the following command to create a new environment from the `requirements.yml` file:
   ```bash
   conda env create -f requirements.yml
   ```

3. **Activate the Environment:** Once the environment is created, activate it using:
   ```bash
   conda activate gpu_env3
   ```

## Usage

To start the training process, navigate to the project root and execute:

```bash
python main.py [OPTIONS]
```

Where `[OPTIONS]` can include:

- `--image_size`: Specify the dimensions of input images.
- `--batch_size`: Define the batch size for training and validation.
- `--num_train_epochs`: Set the number of epochs for training.
- `--initial_learning_rate`: Initial learning rate for optimization.
- `--weight_decay`: Weight decay factor for the optimizer.
- `--early_stopping_patience`: Early stopping criterion based on validation loss.
- `--patience_lr_scheduler`: Patience for learning rate scheduler adjustments.
- `--factor_lr_scheduler`: Multiplicative factor for reducing the learning rate.
- `--model_name`: Path or identifier for the model to be used.
- `--freeze_flag`: Boolean to indicate if the model backbone should be frozen.
- `--data_aug_flag`: Boolean to enable or disable data augmentation.

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

This framework is distributed under the wtfpl license. See `LICENSE.txt` for more information.

## Citing DinoVd'eau

If you find this repository useful, please consider giving a star :star: and citation :fish::

```
@article{XXX,
  title={YYY},
  author={Contini, Matteo},
  journal={ZZZ},
  year={2050}
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
