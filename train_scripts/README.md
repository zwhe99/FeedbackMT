**This directory holds all the model training scripts.**



The naming of the scripts follows this format:

`{method}-data.{data}+llm.{base model}+rm.{reward model}`

**Method**

* `sft`: supervised fine-tuning 
* `cont`: continued training
* `contlora`: continued training with lora
* `raft`: RAFT (Reward rAnked FineTuning)
* `raftplus`: RAFT+
* `mrt`: MRT (minimum risk training)
* `mrtplus`: MRT+



**Data**

Parallel data

* `wmt20-train`: high-resource (2M)
* `wmt20-train-20M`: high-resource (20M)
* `wiki-train`: low-resource (200K)
* `wiki-2_5k-clean`: low-resource (2.5K x 4 language pairs = 10K)
* `wiki-25k-clean`: low-resource (25K x 4 language pairs = 100K)
* `wiki-250k-clean`: low-resource (250K x 4 language pairs = 1M)
* `wiki-750k-clean`: low-resource (750K x 4 language pairs = 3M)



Monlingual data

* `cc-28k`: high-resource monolingual data (3 languages x 28k = 84k)
* `wiki-28k`: low-resource monolingual data (3 languages x 28k = 84k)



**Base model**

* `llama-2-7b-hf`: [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
* `nllb-200-1.3B`: [facebook/nllb-200-1.3B](https://huggingface.co/facebook/nllb-200-1.3B)
* `nllb-200-3.3B`: [facebook/nllb-200-3.3B](https://huggingface.co/facebook/nllb-200-3.3B)
* `nllb-200-600M`: [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
* `random-600M`: same architecture as nllb-200-600M but trained from scratch (random initialisation)



**Reward model**

* `da`: [anon0616/wmt21-comet-qe-da](https://huggingface.co/anon0616/wmt21-comet-qe-da)
* `mqm`: [anon0616/wmt21-comet-qe-mqm](https://huggingface.co/anon0616/wmt21-comet-qe-mqm)
* `unite`: [Unbabel/unite-mup](https://huggingface.co/Unbabel/unite-mup)