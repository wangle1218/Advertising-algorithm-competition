# 2019腾讯广告算法大赛代码说明

## 运行环境

保证电脑内存 16 G 及以上，不过 maxOS 8G 的内存也能跑。以下环境测试过可以运行。

- macOS &#160; &#160; 10.13.6

## 所需依赖包

- anaconda &#160;&#160;&#160;&#160;&#160; 1.6.5
- lightgbm  &#160; &#160; &#160; &#160;   2.0.11
- numpy &#160; &#160; &#160; &#160; &#160; &#160; 1.14.2
- pandas   &#160; &#160; &#160; &#160; &#160;&#160;  0.20.3 
- scikit-learn  &#160;&#160;&#160; &#160;  0.19.0

## 运行说明

在 `src` 同级目录下建立文件夹 `data`，并且将训练集和测试集相关数据放在里面；输入数据文件结构保证如下所示才能成功被读取：

```
data
├── total_data/ (包含所有复赛数据文件)
|   ├── BTest / (复赛B榜测试集数据)
|   |   ├── Btest_sample_bid.out
|   |   ├── Btest_select_request_20190424.out
|   |   └── BTest_tracklog_20190424.txt
|   └── track_log / (包含4月10号到22号的广告竞争队列文件)
|   |   ├── track_log_20190410.out
|   |   ├── track_log_20190411.out
|   |   ├── track_log_20190412.out
|   |   ├── track_log_20190413.out
|   |   ├── track_log_20190414.out
|   |   ├── track_log_20190415.out
|   |   ├── track_log_20190416.out
|   |   ├── track_log_20190417.out
|   |   ├── track_log_20190418.out
|   |   ├── track_log_20190419.out
|   |   ├── track_log_20190420.out
|   |   ├── track_log_20190421.out
|   |   └── track_log_20190422.out
|   ├── final_map_bid_opt.out
|   ├── final_select_test_request.out
|   ├── map_ad_static.out
|   ├── test_sample_bid.out
|   ├── test_tracklog_20190423.last.out
|───└── user_data.out

```

进入项目所在文件及路径，在终端使用 bash 运行 `run.sh` 文件：

```python
cd L.WANG_v2
bash -x run.sh
```

最终得到输出文件 `submission.csv`，存放于 `src` 同级目录下。

同时在`src`文件夹同级目录下建立`tmp_data`文件夹存放模型运行过程中生成的零时文件，整个复现代码的文件结构如下所示：

```
Project
|--src/ (必要代码)
|--data/ (训练、测试数据)
|--tmp_data/ (中间零时文件)
|--models/ (模型结构和参数文件)
|--run.sh (运行脚本)
|--submission.csv（预测的结果文件）
|--README.md
```

## 从原始数据输入到最终结果产出的步骤说明

一、使用脚本`gen_train_data.py` 从原始数据中构造完整的训练集和测试集，具体步骤如下：

1. 从`track_log `中的每天的曝光竞争队列中统计提取每个广告每天的曝光数量；从`Btest_select_request_20190424.out`提取测试当天（4月24号）的广告的覆盖人数信息；从`test_tracklog_20190423.last.out`和`final_select_test_request.out`中提取4月23号的每个广告的覆盖人数信息。
2. 从`final_map_bid_opt.out `中，确定训练样本，与`map_ad_static.out `文件进行拼接，构造完整的训练样本。
3. 使用统计的广告曝光量数据为训练样本匹配标签，并扩充训练样本数量。
4. 拼接`map_ad_static.out `和`Btest_sample_bid.out `生成测试样本。

二、使用脚本`gen_features.py`提取特征。

三、使用脚本`lgb_model.py`训练模型，分别使用全量数据和部分数据进行训练，得到预测输出结果两个。

四、使用规则对上述输出预测结果进行修正，最后将这个结果进行加权融合，得到最后的结果产出。脚本代码为`rule_and_merge.py`。


## 特征的使用和生成情况
主要有三类特征。

一、时间、以及时间序列特征

1. 前两天到前八天的历史曝光，以及这7天历史曝光的统计特征：均值、方差、和、最大值、最小值、计数特征。
2. 上述7天曝光数据的一阶异差特征，以及这些一阶异差的均值、方差、和。
3. 当天和预测那天的广告覆盖范围，广告从覆盖范围中曝光胜出的历史平均胜出概率，缺失值填充为均值；历史平均胜出概率和预测那天的覆盖数量的乘积、与当天妨碍数量的乘积。
4. 广告预测日期距离广告建立日期的天数。

二、排序和计数特征

1. 广告账户id拥有多少个广告id，以及多少个商品id。
2. 广告账户id 中的所有 广告id 的建立顺序，以及所有 商品id 的建立顺序；对于缺失值填充1。

三、组合、以及组合统计特征

1. 将类别特征进行两两组合成组合特征，将组合后的新特征值中频次低于5的，使用 -1 代替。
2. 将上述组合特征和类别特征进行五折交叉统计（将数据分成5份，使用其中4份和训练集标签做统计，将统计结果应用在剩下的一份数据上，可以避免特征过拟合），统计各个特征下的各个特征值的历史平均曝光数量。

所有特征中，对于没有特别说明的缺失值不进行填充。

## 模型的结构定义和参数
使用的开源框架 lightgbm ，参数如下：

```
seed = 2014 + i*12 # 十折，i 为折数，i \in [0,9]
params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'mae'},
        'max_depth':-1,
        'num_leaves':31,
        'min_data_in_leaf':50,
        'learning_rate': 0.025,
        'lambda_l1':6.5,
        'lambda_l2':1.8,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbosity': 0,
        'seed':seed
    }

gbm = lgb.train(params,lgb_train,num_boost_round=10000,)
```
