
## R3S

Users of recommendation systems usually focus on one topic at a time. When finishing reading an item, users may want to access more relevant items related to the last read one as extended reading. 

However, conventional recommendation systems are hard to provide the continuous extended reading function of these relevant items, since the main recommendation results should be diversified. 

In this paper, we propose a new task named recommendation suggestion, which aims to (1) predict whether users want extended reading, and (2) provide appropriate relevant items as suggestions.

These recommended relevant items are arranged in a relevant box and instantly inserted below the clicked item in the main feed. 

The challenge of recommendation suggestion on relevant items is that it should further consider semantic relevance and information gain besides CTR-related factors. Moreover, the real-time relevant box insertion may also harm the overall performance when users do not want extended reading. 

To address these issues, we propose a novel Real-time relevant recommendation suggestion (R3S) framework, which consists of an Item recommender and a Box trigger. We extract features from multiple aspects including feature interaction, semantic similarity and information gain as different experts, and propose a new Multi-critic multi-gate mixture-of-experts (M3oE) strategy to jointly consider different experts with multi-head critics.

In experiments, we conduct both offline and online evaluations on a real-world recommendation system with detailed ablation tests. The significant improvements in item/box related metrics verify the effectiveness of R3S. Moreover, we have deployed R3S on WeChat Top Stories, which affects millions of users. 

### Requirements:
- Python 3.8
- Tensorflow 2.4.1

## Note

In the actual online system, R3S is a complex re-ranking framework implemented in C++. 
All models are trained based on a deeply customized version of distributed tensorflow supporting large-scale sparse features.

Without massive data and machine resources, training R3S is not realistic.

Therefore, the open source code here only implements a simplified version for interested researchers. If there are any errors, please contact me. Thanks!

## About

"Real-time Relevant Recommendation Suggestion" ([WSDM 2021](http://nlp.csai.tsinghua.edu.cn/~xrb/publications/WSDM-21_R3S.pdf))
