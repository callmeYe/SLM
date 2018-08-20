### Segmental Language Models

A Pytorch Implementation.

#### Dependencies

Python 3.6.3 :: Anaconda custom (64-bit)

Pytorch: 0.3.0.post4

Numpy: 1.13.3

Pandas: 0.20.3

#### Setup

For example, if you want to train the model on pku dataset, you should prepare following files in the “data” directory:

##### pku.txt 

unsegmented original training data

##### pku_test.txt

unsegmented original test data

##### pku_test_gold.txt

segmeted data, gold standard for test data

##### _pku.txt 

This file contains the preprocessed training sentences, for example:

附 图 片 <NUM> 张 

##### _pku_test.txt

The same as _pku.txt, but contains the test sentences.

##### supervised_pku.txt

additional supervised data for pku (1024 sentences), for exapmle:

迈 向 | 充 满 | 希 望 | 的 | 新 | 世 纪 | 
附 | 图 片 | <NUM> | 张 | 

#### Pretrained Word Embedding

Put "unigram256.txt" in the "models" directory, you can modify  the number to keep in  consistency to the real word embedding dimension you use.

#### Training

After preparing the data in the "data" directory, just run

```shell
python train.py
```

During the training, the test is also performed.

Better view the training and test results on TensorBoard. The TensorBoard log can be found at "logs" directory.

#### Model Config

Remember to set

```python
DATA = 'pku'
```

In the config.py, other configuration can also be modified in this file.

Set BATCH2 to 0 for unsupervised training.

#### Results

Results can be found at the "results" directory, "result\*" is the original results and we apply the post-processing on "result\*" to get the corresponding "improved_result\*" file.