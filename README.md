
# Vector Quantize PPGs/Bottleneck features

Code for vector quantizing speech dataset, including melspectrograms, phonetic-posteriorgrams/bottleneck features(BNFs). This repo trains an independent module to vector quantize BNFs.

For usage in voice conversion, see [here](https://github.com/warisqr007/vq-ppg-vc)

<!-- ## Block Diagram
![Block Diagram](./block_diagram.jpg)

See details here. [Link](https://anonymousis23.github.io/demos/prosody-accent-conversion/) -->

## Installation
* Install [ffmpeg](https://ffmpeg.org/download.html#get-packages).
* Install [Kaldi](https://github.com/kaldi-asr/kaldi)
* Install [PyKaldi](https://github.com/pykaldi/pykaldi)
* Install packages using environment.yml file.
* Download pretrained [TDNN-F model](https://kaldi-asr.org/models/13/0013_librispeech_v1_chain.tar.gz), extract it, and set `PRETRAIN_ROOT` in `kaldi_scripts/extract_features_kaldi.sh` to the pretrained model directory.


## Dataset

* Acoustic Model: [LibriSpeech](https://www.openslr.org/12). Download pretrained TDNN-F acoustic model [here](https://kaldi-asr.org/models/13/0013_librispeech_v1_chain.tar.gz).
  * You also need to set `KALDI_ROOT` and `PRETRAIN_ROOT` in `kaldi_scripts/extract_features_kaldi.sh` accordingly.
* Vector Quantization:  [[ARCTIC](http://www.festvox.org/cmu_arctic/) and [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/), see [here](https://github.com/warisqr007/vq-bnf) for detailed training process.

All the pretrained the models are available (To be updated) [here](https://drive.google.com/file/d/1RUFXQ9jVXTAgPSukUuWv0TGKGhuaQeeo/view?usp=sharing) 

### Directory layout (Format your dataset to match below)

    datatset_root
    ├── speaker 1
    ├── speaker 2 
    │   ├── wav          # contains all the wav files from speaker 2
    │   └── kaldi        # Kaldi files (auto-generated after running kaldi-scripts
    .
    .
    └── speaker N
    

## Quick Start

See [the inference script](generate_inferences.ipynb)

## Training

* Use Kaldi to extract BNF for individual speakers (Do it for all speakers)
```
./kaldi_scripts/extract_features_kaldi.sh /path/to/speaker
```

* Preprocessing
```
python preprocess_bnfs.py path/to/dataset
python python make_data_all.py  #Edit the file to specify dataset path
```

* Setting Training params.
See conf/

* Training VQ Model
```
./train.sh
```