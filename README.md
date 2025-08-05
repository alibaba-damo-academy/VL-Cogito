<h1 align="center">VL-Cogito</h1>
<p align="center">
<a href="https://github.com/alibaba-damo-academy/VL-Cogito" target="_blank" rel="noopener">Website</a>
&nbsp;&nbsp;
<a href="https://huggingface.co/csyrf/VL-Cogito" target="_blank" rel="noopener"> Model </a>
&nbsp;&nbsp;
<a href="https://huggingface.co/datasets/csyrf/VL-Cogito" target="_blank" rel="noopener"> Dataset </a>
&nbsp;&nbsp;
<a href="https://arxiv.org/abs/2507.22607" target="_blank" rel="noopener">Paper</a>
</p>
The homepage of our multimodal reasoning model—VL-Cogito! 
Inspired by the Latin word “Cogito” (“I think”), VL-Cogito is built for complex and diverse multimodal reasoning tasks, with a strong focus on autonomous thinking and adaptability.

**What makes VL-Cogito stand out?**

Progressive Curriculum Reinforcement Learning (PCuRL):Through a multi-stage, “from easy to hard” reinforcement learning approach, VL-Cogito’s reasoning abilities are significantly enhanced across a wide range of multimodal scenarios!

**Two key innovations:**
+ Online difficulty weighting: Dynamically adjusts training difficulty, allowing the model to progress step by step from easier to more challenging examples.
+ Dynamic length reward: Encourages the model to adapt the length of its reasoning process based on the complexity of each individual problem, balancing both accuracy and efficiency.

**Outstanding Performance:**

VL-Cogito demonstrates stable, state-of-the-art or superior results on mainstream multimodal reasoning benchmarks, covering mathematics, science, logic, and commonsense understanding!

![The framework of our model.](./vl_cogito.png)
