# gptrec-rl

This is an official repository for the paper "Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning", co-authored by [Aleksandr Petrov](https://asash.github.io) and [Craig Macdonald](https://www.dcs.gla.ac.uk/~craigm/)
The paper is accepted for publication at the 2nd Workshop on Recommendation with Generative Models, co-located with the WWW '24 conference. 

If you use this code, please cite:

```
@inproceedings{petrov2024recjpq,
  author = {Petrov, Aleksandr V. and Macdonald, Craig},
  title = {Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning},
  year = {2024},
  booktitle = {The 2nd Workshop on Recommendation with Generative Models, co-located with ACM The Web Conference  2024},
  location = {Singapore, Singapore},
}
```



The code is based on the aprec framework from the bert4rec_repro repo, https://github.com/asash/bert4rec_repro; please follow the original instructions to set the environment. 

Config files for the GPTRec reinforcement learning paper can be found in the configs/generative_rl folder. 
The implementation for the 2-stage pre-training/fine-tuning recommender is in the recommenders/rl_generative folder 


