## More comparison results



**Table 5:** Performance metrics of more models on Assist09 and EdNet datasets.
|    Dataset          | Assist09 |       |             |       |              |       | Ednet    |       |             |       |              |       |
| ------------ | -------- | ----- | ----------- | ----- | ------------ | ----- | -------- | ----- | ----------- | ----- | ------------ | ----- |
|      Bias Type        | baseline |       | plag |       | guess |       | baseline |       | plag |       | guess |       |
|             | auc      | acc   | auc         | acc   | auc          | acc   | auc      | acc   | auc         | acc   | auc          | acc   |
| simpleKT     | 76.90    | 72.50 | 75.60       | 71.94 | 76.19        | 72.90 | 68.93    | 67.34 | 68.17       | 65.73 | 68.28        | 66.51 |
| AKT          | 77.35    | 73.19 | 76.17       | 71.86 | 76.99        | 73.35 | 69.07    | 67.18 | 67.40       | 65.65 | 68.09        | 66.77 |
| CL4KT        | 76.48    | 72.42 | 74.79       | 69.83 | 75.49        | 72.03 | 68.06    | 66.62 | 65.62       | 65.54 | 66.47        | 66.27 |
| DTransformer |    66.12      |  64.96     |     -        |    -   |      -        |    65.37   |     62.18     |   -    |      -       |   -    |      -        |    -   |
| **Ours**         | 81.33    | 75.48 | 80.44       | 73.76 | 0.81         | 74.66 | 76.06    | 70.48 | 74.52       | 68.22 | 75.02        | 69.86 |

Note: We didn't achieve the optimal performance reported in the original paper on our dataset using the default configuration of [DTransformer](https://github.com/yxonic/DTransformer). Due to time constraints, we didn't make further adjustments to the parameters.

## Biased and unbiased knowledge tracing results of one example student 


![](.assets/biased.png)

**Figure 7: Biased knowledge tracing results.**

![](.assets/unbiased.png)

**Figure 8: Unbased knowledge tracing results.**

Our initial approach focused on predicting outcomes based on the question level. To delve into knowledge tracking at a skill level, we opt to average the embeddings of questions associated with each knowledge concept. This averaged representation serves as input to the model, capturing the current knowledge state of the skill.

To illustrate biased and unbiased knowledge tracing outcomes, we analyze a real learning sequence, showcasing our estimation of mastery level evolution for each knowledge concept. Training is conducted under a plagiarism-bias scenario, with evaluations performed on a clean learning sequence.

In the visualizations, circles denote distinct knowledge concepts, with each row depicting the evolution of a specific concept's mastery. A white dot within a circle signifies an incorrect response. Figures 5 and 6 exemplify unbiased and biased knowledge tracing outcomes, respectively, under a plagiarism bias.

Notably, We can observe an increase in the mastery of knowledge states over time, even when encountering incorrect responses, which may result from a high probability of correct answers under plagiarism bias from Figure 5. The unbiased knowledge extractor swiftly adjusts knowledge states in response to errors, underscoring our method's efficacy in adeptly tracking knowledge states, even when trained on biased datasets.