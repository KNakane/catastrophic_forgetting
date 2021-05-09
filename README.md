catastrophic_forgetting
==
# Overview

# Requirements
```
$ pip install -r requirements.txt
```

# Usage
```
python main.py --n_epoch (num of epoch default:10)
               --task_num (num of task default:2)
               --batch_size (batch size default:32)
               --method (default:None)
               --opt (optimizer default:SGD)
               --lr (learning rate default:0.001)
```

# Cite
- [Elastic Weight Consolidation(EWC)](https://arxiv.org/pdf/1612.00796.pdf)
    - [github](https://github.com/yashkant/Elastic-Weight-Consolidation)
- [Online EWC(OnlineEWC)](https://arxiv.org/pdf/1805.06370.pdf)
- [Continual Learning Through Synaptic Intelligence(SI](https://arxiv.org/abs/1703.04200)
    - [github](https://github.com/ganguli-lab/pathint)
- [Learning without Forgetting(LwF)](https://arxiv.org/abs/1606.09282)
- [Continual learning with hypernetworks(HyperNet)](https://arxiv.org/abs/1906.00695)
    - [github](https://github.com/gahaalt/continual-learning-with-hypernets)
- [Continual Learning with Deep Generative Replay](https://arxiv.org/pdf/1705.08690.pdf)
    - [github](https://github.com/kuc2477/pytorch-deep-generative-replay)