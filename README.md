# DACE: Debiased Cognition Representation Learning for Knowledge Tracing
Considering some abnormal behaviors (like guessing and plagiarism) lead to the existence of bias data in students' interaction answer records, we propose the DACE model to learn a robust and debiased representation of knowledge states for reliable prediction on the KT task. 

### Briefly introduce the DACE model
![framework](.assets/framework.png)

Given the processed interaction data, the whole framework mainly contains two modules:
1.  Two knowledge state extractors: Both of them utilize a sequential contrastive learning network to learn representations of knowledge state. However, they are trained differently to capture distinct aspects of the data. 
2. Training process: The biased knowledge state extractor is trained using a combination of contrastive loss and cross-entropy loss, with an early stop mechanism. The unbiased extractor is trained independently by minimizing the information bottleneck loss. 

For more details, please read our paper: "__Debiased Cognition Representation Learning for Knowledge Tracing__"

### How to train the DACE model
1. Data Preparation: 
   - Download and unzip [assist09 data](https://drive.google.com/uc?export=download&id=14wBw8BHf9e328v4dD5EdsRMtR_gCFcdq) in `data` directory. 
   - Download and unzip [question embedding](https://drive.google.com/uc?export=download&id=16s9jNZZSkxT33Hb7r1OY7PqkJBzhC5DV) in `embeddings` directory.
2. Bias Injection: 
   - Modify the `config.py` by specifying the `biased types` (e.g., plagiarism, guessing) and adjusting the `injection proportion` and `p`.
3. Main Files: 
   - Run the following command: 
        ```
        python train.py # pretrain sequence model on biased dataset
        python run.py # train on biased dataset and evaluate on clean dataset
        ```

Please note that, this model is also suitable for regular KT prediction. If you want to test the performance on normal KT datasets, simply set the `biased type` to `None`.


----

### Dataset BaiPY

The [BaiPY](https://drive.google.com/file/d/1Zjkmc9w-LJIRfwlQAeq3vPw9Bxml_jJf/view) dataset has been collected from an automatic program evaluation and open teaching assistance platform for universities. The platform offers a wide range of courses, attracting millions of registered students. For this study, the BaiPY dataset specifically focuses on the Python course and online programming instruction. To provide a comprehensive analysis, we will primarily compare the BaiPY dataset with non-programming datasets and programming datasets utilized in knowledge tracing tasks.

### Comparison with Non-programming Datasets

The detailed statistics presented in Table [Dataset Statistics](#1) indicate that the BaiPY dataset consists of 110,909 students, 1,149 questions, 52 knowledge concepts, and over 420 million interaction records. This highlights a notable advantage of the BaiPY dataset, as it boasts the largest number of students and records, allowing for exploration of knowledge tracing tasks on a significantly larger scale. Furthermore, the BaiPY includes information that is not available in public datasets, such as question difficulty, hierarchical relationships between knowledge concepts, and question content.

### Comparison with Programming Datasets

The comparison of BaiPY with other public programming datasets is shown in Table [Dataset Comparison](#4). The table mainly compares the programming language (i.e., Lang), data source, number of students (\#students), and four types of information (i.e., submitted code, click events, problem content, and knowledge concepts). As depicted in Table [Dataset Comparison](#4), BePKT is a comprehensive dataset based on C/C++ language. In terms of the Python programming language, our BaiPY dataset is currently the largest dataset with the largest number of students for online judging when compared to other Python-based datasets such as CloudCoder and CodeBench. Moreover, the BaiPY dataset contains abundant information such as question contents (Question), code contents from students' feedback (Code), historical click events (Click Event), and the knowledge concepts associated with each question (concepts). This rich set of information provides an excellent environment for promoting research in this field.

### Detailed Information

The dataset encompasses various question types such as multiple-choice, true-false, and programming questions. We present an example of [Sample Question](#2) and [Sample Code](#3), which depict a programming question and a code snippet from a student. The programming question in [Sample Question](#2) follows a format commonly found on online judge (OJ) platforms. It includes a problem statement, along with sample input and output data. Contrary to the single feedback provided for multiple-choice or true-false questions, programming questions offer specific feedback codes that can effectively diagnose a student's knowledge state. This feature showcases the added value of our newly released datasets.

Additionally, the online judging system automatically records each student's submission upon completion. To demonstrate this, we present an example of a submission record in [Submission Record](#5), which reflects the click events captured in the BaiPY dataset. The record includes details such as the student's ID (`User ID`), the problem attempted (`Problem ID`), the associated knowledge skill (`Knowledge Skill`), and the submission time (`Create At`). Furthermore, it provides the final score (`score`) and success status (`success`). Notably, it even includes the score and time taken for each test case in the `Judge Response` section. With such comprehensive information on click events, we believe that the release of BaiPY will be beneficial to the research community and attract greater attention to the topic.


#### <span id = "1">Dataset Statistics</span>
| Datasets  | Assist09 | Assist12 | EdNet | BaiPY   |
|-----------|----------|----------|-------|---------|
| \# Students | 3,852    | 27,485   | 5,000 | 110,909 |
| \# Questions($N_Q$)  | 17,737   | 53,070   | 12,103 | 1,149   |
| \# concepts($N_S$)      | 123      | 265      | 188    | 52      |
| \# Records($N_R$)     | 282,619  | 2,709,647 | 644,682 | 4,271,167 |

#### <span id = "4">Dataset Comparison</span>
| Dataset    | Lang   | Source        | \#students | Code | Click Event | Question | Skills |
|------------|--------|---------------|------------|------|-------------|----------|--------|
| PLAGIARISM | C/C++  | ide           | N/A        | &checkmark; | &cross; | $\times$ | $\times$ |
| BlackBox   | Java   | ide           | 1M         | N/A  | N/A         | N/A      | N/A    |
| CloudCoder | Python/C | online ide  | 646        | N/A  | N/A         | N/A      | N/A    |
| Code.org   | Scratch | N/A           | 500K       | N/A  | N/A         | $\checkmark$ | N/A |
| POJ        | C/C++  | online judge  | 104        | $\times$ | $\times$ | $\times$ | $\times$ |
| CodeHunt   | Java/C# | online ide   | 258        | $\checkmark$ | $\checkmark$ | $\times$ | $\times$ |
| CodeBench  | Python | oneline judge | 2714       | $\checkmark$ | $\checkmark$ | $\times$ | $\times$ |
| BePKT      | C/C++  | online judge  | 906        | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| **BaiPY**  | Python | online judge  | 110,909    | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |


#### <span id = "2">Sample Question</span>

```python
# This question requires statistics of the integer with the most number of occurrences in an integer sequence and its number of occurrences.
# Sample Input: `10 3 2 -1 5 3 4 3 0 3 2' 
# Sample Output: `3 4'
```


#### <span id = "3">Sample Code</span>

```python
string = list(input().split())
nums, amount, max_count = string[1:], int(string[0]), 0
for i in range(amount): 
    if(nums.count(nums[i]) > max_count):
        max_count = nums.count(nums[i])
        ret = i
        print("%s %s"%(nums[ret], nums.count(nums[ret])))
```


#### <span id = "5">Submission Record</span>

```json
{
    "Submission ID": "14967425",
    "User ID": "7fb6e8f8",
    "Create At": "2022-02-24 15:03:31",
    "Problem ID": "14963215",
    "Knowledge Skill": "Python loops",
    "Judge Response": {
        "testcaseJudgeResults": {
            "0": {
                "testcaseScore": 5,
                "time": 0.002,
            },
            "1": {
                "testcaseScore": 5,
                "time": 0.002,
            },
            "2": {
                "testcaseScore": 5,
                "time": 0.002,

            }
        },
        "score": 15,
        "success": true
    }
}
```

