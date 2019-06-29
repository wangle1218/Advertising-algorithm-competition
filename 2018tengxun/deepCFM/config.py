
# set the path-to-files
TRAIN_FILE = "../data/combine/merge_train1.txt"
TEST_FILE = "../data/combine/merge_test1.txt"

SUB_DIR = "./output"

MAXLEN = 6
NUM_SPLITS = 20
RANDOM_SEED = 2018

# types of columns of the dataset dataframe
CATEGORICAL_COLS = [
    'advertiserId','campaignId','creativeSize',
    'adCategoryId','productType','productId',
    'aid','LBS','age','consumptionAbility',
    'house','education','gender','os','ct',
    'marriageStatus','house','carrier'
]

# NUMERIC_COLS = [
#     'advertiserId_LBS_stas', 'campaignId_LBS_stas', 'adCategoryId_LBS_stas',
#     'advertiserId_age_stas', 'campaignId_age_stas', 'adCategoryId_age_stas',
#     'advertiserId_consumptionAbility_stas', 'campaignId_consumptionAbility_stas', 'adCategoryId_consumptionAbility_stas',
#     'advertiserId_education_stas', 'campaignId_education_stas', 'adCategoryId_education_stas',
#     'advertiserId_gender_stas', 'campaignId_gender_stas', 'adCategoryId_gender_stas',
# ]

NUMERIC_COLS = []

MULTI_VALUE_COLS = [
    'topic1','topic2','interest1',
    'interest3','interest4','kw3','topic3',
    'interest2','interest5','kw1','kw2'
]

IGNORE_COLS = [
    'appIdAction', 'appIdInstall','creativeId',
    # 'interest3','interest4','kw3','topic3'
]
