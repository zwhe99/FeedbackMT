# FeedbackMT


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
    --model-type s2s \ # `--model-type s2s` for NLLB; `--model-type causal` for LLAMA-2
    --beam 4
```
