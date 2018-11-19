# 2018IJCAI-阿里妈妈广告搜索转化预测大赛/腾讯广告算法大赛总结

前段时间参加了阿里妈妈和腾讯的两个广告算法竞赛，一直没时间做个总结，今天抽时间稍微整理一下广告算法竞赛的一些思路以及参赛的心路历程。阿里妈妈的竞赛其实持续了很长时间，但是由于时间关系，初赛基本没怎么做，在初赛最后5天草草写了份代码，勉强进入复赛；而进入复赛后又是在进行腾讯赛的间隙抽时间做的，做了10天，期间还有两次因为提交结果文件的格式问题，线上评测失败了。而腾讯赛又是在写论文的间隙做的。本来没怎么接触过广告算法之类的模型，又没时间去看论文，所以非常遗憾的，两个比赛名次都比较靠后，好在参赛人数比较多，最后勉强维持在 TOP4%-5% 的名次的样子。

由于这两个比赛的数据量都是非常大的（阿里的数据在10G的样子，腾讯的数据在16G的样子），而我的笔记本内存却只有8G，可以说这两个比赛做的非常辛苦。本文就简单从大数据处理、特征工程和模型的选择几个方面分别对两个比赛进行总结。

## 1、小内存电脑如何处理大数据

### 数据加载

对于非常大的数据，直接全部加载到内存是不可能的了，比如阿里的数据解压后10个G，这是数据都是原始数据，里面包含用户个人信息、商铺信息、广告信息、用户的搜索行为数据（其实原始文件就是这几个表拼接成的一个大表）。既然不能全部加载到内存对数据进行处理，那么我们自然可以选择分而治之的想法，将这个大表拆分成几个小表，然后分别对用户、商铺、广告和搜索行为进行特征提取，以及其他数据预处理的过程。我使用了下面的代码将大表拆分：

```python
user_feat = ['user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level']
reader = pd.read_csv("./data/round2_train.txt", sep="\s+",iterator=True)
chunks = []
loop = True
while loop:
    try:
        chunk = reader.get_chunk(500000)[user_feat]
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print("Iteration is stopped")
df_user = pd.concat(chunks,axis=0, ignore_index=True)

df_user = pd.concat(chunks,axis=0, ignore_index=True)
df_user = pd.concat([df_user, test[user_feat]],axis=0)
df_user.drop_duplicates(subset='user_id',keep='first',inplace=True)
df_user.to_csv('./data/user_file.csv',index=False)
print('user_file', df_user.shape)

del df_user
gc.collect()

```

整个表有非常多列，非常多行，我们使用`pandas`的`iterator=True`属性迭代的读取数据块，单独将用户属性列`user_feat`抽取出来，每次读取500000行。这样最后我们将所有用户数据进行去重处理，然后保存。这样几千万行的用户数据就剩下几十万行了。

其他几个小表也这样操作拆分。另外，在进行数据分析的时候，我发现店铺数据它是随着时间变化有变化的，比如同一个店铺，在不同的日子，它的`shop_review_num_level`，
`shop_review_positive_rate`，`shop_star_level`，`shop_score_service`，
`shop_score_delivery`，`shop_score_description`几个属性的值是变化的，所以对于店铺表进行拆分时，我们需要将日期也拆出来，去重的时候也要根据日期去重，这样才不至于丢失一些信息（其实这个发现也有助于后面的特征工程）。

这样，在后期我们就可以分别提取特征，然后将特征文件进行拼接，训练了。

在腾讯赛中，广告信息、用户信息、训练数据都是分开的，另外光用户数据就有16G大，全部加载到内存进行 `merge`操作是不可能的了，因此，我在初赛和复赛期间都是将所有数据划分成好多小份来处理、训练的（当然这也一定程度上影响了模型的性能）。怎么划分呢，因为赛题是相似用户拓展，目的是对于某个广告，根据点击过这些广告的人去的特征取找相似特征的人群。因此我选择了将所有数据按照广告的种子包（一个广告id对应一个种子包）来划分，这样最后4700多万的用户数量被分散到20几个小文件之后变成了5000多万用户（有重复的，这么多重复的用户在复赛竟然可以构造一个强特征）。按种子包拆分用户文件的代码如下：

```python
seed = pd.DataFrame(data['aid'].value_counts())
seed['aid1'] = seed.index
seed.columns = ['num','aid1']

seed = seed.reset_index(drop=True)
num_cut = []
n = 1
num = 0
for i in range(seed.shape[0]):
    num += seed.loc[i,'num']
    if num > 3500000:
        num = seed.loc[i,'num']
        n += 1
    num_cut.append(n)
    
seed['num_cut'] = num_cut

aid_group = {}
group = seed.num_cut.values
aid = seed.aid1.values

for i in range(len(aid)):
    if group[i] not in aid_group.keys():
        aid_group[group[i]] = [aid[i]]
    else:
        aid_group[group[i]].append(aid[i])

pickle.dump(aid_group,open('./data/aid_group.pkl','wb'))

for aid_id in aid_group.keys():
    aid_list = aid_group[aid_id]
    aid_uid = data.loc[data['aid'].isin(aid_list),'uid'].unique()
    pickle.dump(aid_uid,open('./data/aid_uid_%s.pkl' % aid_id, 'wb'))

```
根据训练集和测试集中的广告id和用户id的对应关系，将同一个广告id对应的所有用户id保存在一起。根据用户数量确定每个文件存多少用户（代码里是350W）。

### 写入磁盘

在构造完特征，对所有文件进行拼接之后，我们可以将生产的整个训练数据保存下来，以供下次进行调参训练，这样就不需要在每次训练之前都花时间来提取特征。但是由于数据量大，提取完的训练文件更大，如何快速的、又省内存的保存到磁盘，也是需要考虑的事情。一般我习惯使用`pickle`模块将数据保存，但是当数据大小超过2G左右就不管用了， pickle 根本不能成功将数据 dump。而使用 csv 文件来保存的话，`pd.to_csv()`将数据写入磁盘的时间将非常非常长。。。所以这不是一个好选择。

这里，我们首先给数据进行瘦身，比如将`float64`格式改成`float16`，将`int64`改成`int16`或者`int8`。这样节省不少内存，同时使用`pd.to_csv()`保存数据时，进行选择使用`txt`格式，而不用`csv`格式，这样数据IO速度快很多

```python
pd.to_csv('data.txt',index=False)
```
另外也可以使用`hdf5`文件格式存储数据。但是，如果你启用了虚拟内存，当数据占满了内存时，你把数据写入磁盘就基本写不动了，这时，我们可以选择逐行写入txt文件

```python
//ftrain =  open(fileOUT_path,'w')
ftrain =  open(fileOUT_path,'a+')

for i in range(num):
	...
	...
	
ftrain.write(data_dict['label'][i] + ' ' + ' '.join(feats) + '\n')
ftrain.close()
```

但是要注意，不要用'w'模式，在腾讯赛初赛时，我将训练数据加载到内训，提完特征，处理ffm数据格式完后，将数据写入磁盘文件，用了`ftrain =  open(fileOUT_path,'w')`，我深深的记得大概花了30个小时，数据才写完。后来在复赛的时候，我把'w'改成'a+'，于是变得非常快，3个小时左右写完之前需要30小时才能写完的数据量（这一心痛的经历如果有个大内存，就不会承受了。。）。

## 2、特征工程

### IJCAI阿里妈妈

- 交叉统计特征

对于购物来说，不同的性别、年龄、职业可能会对不同的商品品牌、商品价格等属性感兴趣，因此我们可以考虑构造这些交叉特征的转化率特征来刻画这种现象。在代码文件`static_features.py`中，我将提取这类特征封装成一个类，可以直接使用。在计算转化率时有些特征组合出现的频次少，我直接将这些频次少的组合进行了过滤，以防止过拟合。另外计算训练集的转化率时，为了防止数据泄露，我采用了五折交叉的方式，将所有数据分成5份，使用其中的4份来计算剩下的那份数据的转化率。

- 组合特征

我们可以想到，同样性别的用户，如果年龄不同、职业不同、消费水平不同，等等，那他们的个人属性肯定也会有区别。所以有必要对这些特征进行组合，比如性别为`1`，年龄为`3`，那么通过组合，我们可以得到一个`性别-年龄`的特征`1_3`。如果使用线性模型，这样做显然是有用的，这样等于在one-hot之后加入了一个非线性特征，但是对树形模型有没有用呢，毕竟树形模型能在训练时学习到非线性特征，我觉得还是有用的，因为树形模型在训练时对特征的组合具有随机性，通过手动构造这样的组合特征等于我们直接的指定要模型学习这种组合特征。

- 店铺特征

我们知道淘宝店铺有店铺等级、店铺好评率、发货速度评分、商品描述评分等属性，通过分析，我发现几乎大部分店铺的评分非常接近都在，都在同一水平线附近，可以说区分性非常小，为了让所有店铺的这些属性有区分性，我们可以将所有评分进行分段离散化（将店铺在每个属性维度上分等级），但是分多少段不好确定；另外也可以使用`StandardScaler()`方法将所有评分进行标准化处理，这样使得所有评分有了较好的区分性，同时也保持了评分的连续性。

另外，在前面我说了，每个店铺在不同日子的各属性评分是有变化的，于是我们可以统计一下每个店铺在几天内的这些属性评分的变化情况，比如方差、均值、总和等，这样其实侧面刻画了每个店铺在每天的销售变化情况，因为只有店铺销售了商品，才能有新用户给店铺打分。

- 用户搜索行为特征

在复赛中，由于‘7号’是双十一活动，流量分布和平时不同，因此，对于用户搜索行为特征的构造，我分成了如下三个时间段的提取，31至5号，6号，7号上午（7号下午是测试数据）。统计31至5号所有用户的搜索次数均值、方差、总和，使用用户6号的搜索次数减去上述用户的搜索均值，由于6号应该正是双十一的前一天，可能有大量用户进行囤货之类的操作，等双十一下单。对广告商品做同样的特征提取，可以描述广告商品的曝光率等信息，也能从侧面反映出哪些商品是热销商品。

由于三个时间段的流量分布非常不同，广告的搜索转化率也相差很大，而要预测的又是7号下午的转化率情况，因此我选择只利用前两个时间段的数据来提特征，而第三个时间段的数据利用来提取更加细化的特征，并且将其作为训练数据来构造线下训练集和验证集。因为在复赛结束还剩10天的时候我才做，第一次提交结果就到了前100名让我非常惊讶，我以为500个进入复赛的人没多少人在做了，后来在群里交流了下才知道非常多的人使用了全量的数据来训练，这样导致导致他们的训练集和线上测试集的数据分布非常不一致，那天我在钉钉群里爆出这个的时候非常多的人加我好友咨询....可相知而正确利用数据构造训练集是多么重要（这需要仔细的分析数据，并且多数据具有很好的理解力）。

针对7号上午的搜索行为数据，我进行了更细致的特征，在模型中，重要性都比较排前。用户在某商品上的停留时长`diff_time`，用户搜索广告商品后，点击的商品所在页面和首页相差多少业`diff_page`,在用户搜索点击商品时，某个商品是其第几次点击的商品`user_click_times_k`，统计当天所有商品被搜索的次数`item_look_count`，累计被浏览时长`item_look_time`，商品平均被浏览时长`item_look_per_time`，用户搜索商品后翻页时间和页数的比值`duration`；用户是否对某个店铺的商品有多次点击，某店铺是用户点击的第几个店铺`click_shop_k`，某商品是用户点击的第几个商品`u_click_item_k`，用户点击完某个商品后，在之后还会点击几次商品`click_shop_k_last`，这个应该是个leak特征；根据给出的数据中的搜索商品类别、属性预测属性`predict_category_property`计算预测的商品属性和商品本身所具有的属性数量之间的比值，我称为商品属性预测命中率特征`predict_hit`。以上特征的构造详情可参考代码。

- 时间特征

就是各种特征的时间差分、滑窗统计，不一一细说了。

### 腾讯赛

腾讯赛的特征提取思路比较简单，也是一些交叉统计算转化率，组合特征之类的。但是因为数据量特别大，不大好处理，非常耗时又对内存需求很大。除了上述两组特征外，针对多值离散类的特征如`interest 1、2、3、4、5`，`topic 1、2、2`，`kw 1、2、3`三类11个filed特征，特征取值有几十万个，初赛有开源的代码将这些特征做 bag of word 处理，但这需要非常大的内存，小内存电脑肯定使用不了。在初赛我主要是提取了SVD矩阵分解特征，词向量的embedding特征，做法如下：

```python
tfidf = CountVectorizer(ngram_range=(1, 1), max_df=0.8, min_df=5)
normalizer = Normalizer(copy=False)
for i,feature in enumerate(text_col):
    df_feat = df_text[['uid',feature]].copy()
    df_feat = remove_nonfeat(df_feat,feature,i)
    df_feat = df_feat.fillna(' ')
    X = tfidf.fit_transform(df_feat[feature])
    print('X.shape:',X.shape)
    df_feat = df_feat['uid']

    if feature in big_dim:
        n_components = 55
    else:
        n_components =10# int(0.35*(X.shape[1]))
    svd = TruncatedSVD(n_components)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
    X = np.round(X,6)
    X = pd.DataFrame(X)
    X.columns = [feature + '_lsa' + str(i) for i in range(X.shape[1])]
    X = X.astype(np.float32)
    df_feat = pd.concat([df_feat,X],axis=1)

    df = pd.merge(df, df_feat ,how='left', on='uid')
```

```python
def get_w2v(text,model,feat_dict):
    text = text.split()
    text = [w for w in text if w in feat_dict.keys()]
    w2v = np.zeros(50)
    for word in text:
        try:
            w2v += model[word]
        except:
            continue
    if len(text) == 0:
        return w2v
    else:
        return w2v/len(text)

def gen_w2v_feat(df):
    w2v_feat = ['kw1','kw2','topic1','topic2']

    df_w2v = df[['uid'] + w2v_feat]
    df = df['uid']
    for feat in w2v_feat:
        feat_dict = pickle.load(open('./data/posfeat/%s.pkl' % feat, 'rb'))
        w2v_mar = []
        model = fasttext.load_model('./w2v/%s-model.bin' % feat)
        for row in df_w2v[feat]:
            w2v_mar.append(get_w2v(str(row),model,feat_dict))

        w2v_mar = np.round(np.array(w2v_mar),6)
        w2v_mar = pd.DataFrame(w2v_mar)
        w2v_mar.columns = [feat + '_w2v_' + str(i) for i in range(50)]
        w2v_mar[feat + '_w2v_sum'] = w2v_mar.sum(1)
        w2v_mar[feat + '_w2v_mean'] = w2v_mar.mean(1)
        w2v_mar = w2v_mar.astype(np.float32)
        df = pd.concat([df, w2v_mar], axis=1)

        print(feat,"w2v done!")
    del model
    gc.collect()

    return df
```
其实embedding特征可以不做，直接全部用SVD做分解，但是由于 kw,topic两类特征的取值非常多加起来几十万，也就是做BOW时有几十万维，这样做SVD分解非常慢，也需要大内存，所以没办法只能用embedding，思路就是给每个特征取值训练词向量，然后对filed的对个特征值的词向量取平均。这样操作在初赛时线上有0.7469的成绩，融合一个FFM模型可以到0.749，勉强在进入初赛前50。然而在复赛时数据量增加了4倍多，把数据分种子包分批训练时内存也很吃力，只能进行采样训练，所以到复赛时这样提特征使用LGB来训练就非常不理想。

另外由于初赛和复赛的数据量有非常大的差别，同时训练集出现了很多重复用户（初赛数据训练集很少重复用户），而这些重复用户又是广告的点击转化率非常高的用户，很多人发现提取用户转化率特征可以让模型提高一个百分点。。非常遗憾的我以为复赛数据和初赛数据是差不多的，于是我并没有对复赛数据进行探索分析，所以在复赛时，我和初赛一样将用户uid直接去掉，也没提用户转化率特征。。再一次说明对数据的探索分析是多么的重要。

## 3、模型的选择

阿里妈妈的比赛侧重特征的提取，模型倒不是关键。我分别使用了lgb和xgb，但是由于xgb训练非常慢，没有时间进行五折交叉训练，所以线下成绩比lgb低一些，本打算融合一下lgb和xgb的结果，但是比赛倒数第二天提交融合结果时，第二天一看成绩，，，提交格式错误，没评分。。所以最终也不知道融合提了多少分，能不能提分。。在最后一天由于lgb线下有一些提升，所以最后一次提交预测结果就是用的lgb单模型的结果，没有进行融合，非常遗憾。

腾讯赛非常吃模型，基本上是深度学习模型的天下了，然而在复赛时两大佬直接开源模型，非常多的有服务器的人就靠着开源过日子了。。然而我的电脑却并跑不动开源的模型，这真是一件非常sad的事，因为我看到有人跑开源跑到前20名去了。。。复赛时我在deepffm模型的基础上做了一些修改，针对多值离散特征群,kw,interest,topic特征做了两层卷积，卷积操作需要训练的参数还是少一点的，分种子包训练时我的电脑还能跑的起来，然而我的mac笔记本并没有GPU，训练完估计得几天，所以抽样跑了一下，线下大概比lgb高一个点。后来找到一个有好机器的群友，让他用全量帮我跑了一下这个模型，跑了两个epoch，他说只有0.72，非常失望。。我想应该是参数没调的缘故，比如interest类特征，每个用户的兴趣爱好数量是不同的，有的几个，有的几十个爱好，但是我做卷积操作时为了节省内存，把这些特征统一截断成长度为6，可以说损失了非常多的信息。但是最终我还是不知道是不是参数设置的问题。。。

## 4、总结

这两个比赛做的非常郁闷心酸，没拿到好名次还花费了很多时间精力，而且在做腾讯赛时，由于频繁的大量的数据IO操作差点把磁盘都搞坏了（不知道是不是这个原因），幸好最后通过mac的磁盘修复功能就磁盘修复了。。有兴趣的话可以去看看源代码，但是我的代码并没有写注释或者整理什么的，看明白可能需要一些耐心。