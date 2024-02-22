# FeedbackMT

Code of _[Improving Machine Translation with Human Feedback: An Exploration of Quality Estimation as a Reward Model](https://arxiv.org/abs/2401.12873)_

### Environment

```shell
conda create -n feedbackmt python=3.10.0
conda activate feedbackmt
cd src/LMFlow
pip3 install -e .
cd ../../
pip3 install -r requirements.txt
conda install mpi4py
```



### Data

Download from [Google drive](https://drive.google.com/file/d/1eF6fQKaKl-PcWrV9arYbGyOvRGZrThNb/view?usp=sharing).

```shell
unzip data.zip
```



### Training

See `training_scripts`.



### Inference

```shell
python3 src/inference_sft.py \
    --model-name-or-path <model path> \
    --inst-file data/instruct_follow.txt \
    --lang-pair en-zh \
    --input-file <input file> \
    --output-file <output file> \
    --search-algorithm beam \
    --batch 2 \
    --seed 0 \
    --model-type s2s \ # --model-type s2s for NLLB; --model-type causal for LLAMA-2
    --beam 4
```



### Critical code

```shell
src/LMFlow/src/lmflow/pipeline/raft_aligner.py     # RAFT/RAFT+ for LLAMA2
src/LMFlow/src/lmflow/pipeline/raft_aligner_t2t.py # RAFT/RAFT+ for NLLB
src/LMFlow/src/lmflow/pipeline/mrt_aligner_t2t.py  # MRT/MRT+   for NLLB
```



### Acknowledgement

* [OptimalScale/LMFlow](https://github.com/OptimalScale/LMFlow): The RAFT implementation is based on `LMFlow`.
* [wxjiao/ParroT](https://github.com/wxjiao/ParroT): Training and inference scripts are based on `ParroT`
