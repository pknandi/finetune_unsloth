## Steps - Installation
```bash 
sudo apt-get update
```

```bash 
sudo apt install pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
```
```bash
sudo apt install conda-forge ffmpeg av cython spacy thinc
# for conda environment, 
conda install -c conda-forge ffmpeg av cython spacy thinc
```
```bash
pip3 install audiocraft --no-deps
```
```bash
pip3 install -r requirements.txt
```

## Create Tokenizer
### create csv file for tokenizer dataset
* **first update the dataset dir in the script, then run the command** 
```bash
python3 dataset_to_csv.py
```
### create motion token codebook
* **Update the configurations in the script first. Then run**
```bash
python3 k_means_motion_tokenizer.py
```

## Finetune
### update dataset dir and create dataset mapper for training data
```bash
python3 dataset_to_csv.py
```
### start fine tuning
* **Uncomment build_joint_jsonl, comment out finetune, run once to process training data**
```bash
python3 speech_to_motion_pipeline.py
```
* **Comment out build_joint_jsonl, uncomment finetune, run again to start finetuning**
```bash
python3 speech_to_motion_pipeline.py
```

## Inference and 3D Motion Visualization
### Run this script for inference
* **Update audio_path, finetuned model_dir, tokenizer path**
```bash
python3 run_inference.py
```

### Visualizing Motion
* **Download the SMPL-X Body Models**
  * Go to the official SMPL-X website: https://smpl-x.is.tue.mpg.de/
  * Register for an account
  * Under the Downloads section, download the **SMPL-X v1.1** model files
  * Extract that zip file. You will see a folder named models/smplx. Move that entire models folder into the root directory of the repo.
* **Install Rendering Libraries**
```bash
pip3 install smplx trimesh matplotlib
```
* **Run this Script for output**
```bash
python3 visualize_motion.py
```
It generates an output mp4 video of the skeletal joints moving.

It exports the first 100 frames as standard .obj 3D meshes inside a folder. You can drag and drop these .obj files into any 3D software (like Blender or an online 3D viewer) to see the high-quality human mesh.