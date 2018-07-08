# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import datetime
import time
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")
start = datetime.datetime.now()
##loading data
#test
test_consumer_A = pd.read_csv("./test/scene_A/test_consumer_A.csv")
test_consumer_B = pd.read_csv("./test/scene_B/test_consumer_B.csv")
test_behavior_A = pd.read_csv("./test/scene_A/test_behavior_A.csv")
test_behavior_B = pd.read_csv("./test/scene_B/test_behavior_B.csv")
test_ccx_A = pd.read_csv("./test/scene_A/test_ccx_A.csv")
# train
train_consumer_A = pd.read_csv("./train/scene_A/train_consumer_A.csv")
# train_consumer_B = pd.read_csv("./train/scene_B/train_consumer_B.csv")
train_behavior_A = pd.read_csv("./train/scene_A/train_behavior_A.csv")
# train_behavior_B = pd.read_csv("./train/scene_B/train_behavior_B.csv")
train_ccx_A = pd.read_csv("./train/scene_A/train_ccx_A.csv")
target = pd.read_csv("./train/scene_A/train_target_A.csv")
##task_A
train_consumer_A = train_consumer_A[(train_consumer_A.ccx_id != 1143)]
train_consumer_A = train_consumer_A[(train_consumer_A.ccx_id != 25311)]


dealed_col = {'ccx_id': [1.0, 21245],
              'var1': [1.0, 39],
              
              'var11': [0.2746528594963521, 9],
              'var1163': [0.22720640150623675, 98],
              'var1164': [0.19764650506001413, 100],
              'var1165': [0.16705107084019769, 98],
              'var1166': [0.20221228524358673, 38],
              'var1167': [0.175664862320546, 40],
              'var1169': [0.15165921393269005, 88],
              'var12': [0.2722522946575665, 14],
              'var1271': [0.28722052247587665, 5575],
              'var1272': [0.26297952459402213, 5017],
              'var1273': [0.24160979053895035, 4550],
              'var1274': [0.2981878088962109, 1093],
              'var1275': [0.2744645799011532, 1015],
              'var1276': [0.25445987291127325, 902],
              'var1277': [0.2981878088962109, 802],
              'var1278': [0.2744645799011532, 751],
              'var1279': [0.25445987291127325, 641],
              'var1280': [0.2981878088962109, 2915],
              'var1281': [0.2744645799011532, 2563],
              'var1282': [0.25445987291127325, 2292],
              'var1283': [0.2981878088962109, 52],
              'var1284': [0.2744645799011532, 51],
              'var1285': [0.25445987291127325, 47],
              'var1286': [0.2981878088962109, 80],
              'var1287': [0.2744645799011532, 78],
              'var1288': [0.25445987291127325, 77],
              'var1289': [0.25045893151329723, 24],
              'var1290': [0.2257472346434455, 28],
              'var1291': [0.20145916686279125, 31],
              'var1292': [0.2981878088962109, 2653],
              'var1293': [0.2744645799011532, 2304],
              'var1294': [0.25445987291127325, 2050],
              'var1295': [0.2114850553071311, 3567],
              'var1296': [0.1812661802777124, 2997],
              'var1297': [0.16738056013179572, 2759],
              'var1298': [0.2981878088962109, 1286],
              'var1299': [0.2744645799011532, 1209],
              'var13': [0.2746528594963521, 1999],
              'var1300': [0.25445987291127325, 1110],
              'var1301': [0.25045893151329723, 213],
              'var1302': [0.2257472346434455, 178],
              'var1303': [0.20145916686279125, 129],
              'var1304': [0.2981878088962109, 32],
              'var1305': [0.2744645799011532, 32],
              'var1306': [0.25445987291127325, 32],
              'var1307': [0.2981878088962109, 238],
              'var1308': [0.2744645799011532, 236],
              'var1309': [0.25445987291127325, 226],
              'var1310': [0.2671216756883973, 4986],
              'var1311': [0.23713815015297718, 4343],
              'var1312': [0.21986349729348081, 3949],
              'var1382': [0.17947752412332313, 4],
              'var1383': [0.15457754765827253, 4],
              'var1451': [0.19552835961402684, 37],
              'var1452': [0.16912214638738526, 40],
              'var1454': [0.15768416097905388, 15],
              'var1505': [0.1622028712638268, 41],
              'var155': [0.24862320546010827, 252],
              'var156': [0.2142151094375147, 238],
              'var157': [0.18079548128971523, 228],
              'var158': [0.38329018592610026, 25],
              'var159': [0.38329018592610026, 20],
              'var16': [0.2553542009884679, 28],
              'var160': [0.38329018592610026, 19],
              'var161': [0.38329018592610026, 18],
              'var1610': [0.2230171805130619, 40],
              'var1611': [0.19482231113203108, 42],
              'var1612': [0.16370910802541774, 42],
              'var1613': [0.1986820428336079, 18],
              'var1614': [0.17284066839256296, 17],
              'var1619': [0.24010355377735937, 58],
              'var162': [0.38329018592610026, 19],
              'var1620': [0.20720169451635678, 60],
              'var1621': [0.175335373028948, 59],
              'var1622': [0.2240997881854554, 28],
              'var1623': [0.19228053659684632, 27],
              'var1624': [0.1624852906566251, 27],
              'var163': [0.38329018592610026, 19],
              'var1637': [0.3678983290185926, 3],
              'var1638': [0.363662038126618, 5],
              'var1639': [0.363662038126618, 14],
              'var1640': [0.363662038126618, 9],
              'var1641': [0.363662038126618, 9],
              'var1642': [0.3678983290185926, 25],
              'var1643': [0.363662038126618, 79],
              'var1644': [0.363662038126618, 82],
              'var1645': [0.363662038126618, 85],
              'var1646': [0.363662038126618, 88],
              'var1665': [0.17726523887973641, 35],
              'var1666': [0.15363614968227818, 38],
              'var17': [0.2746528594963521, 20],
              'var1734': [0.24730524829371617, 47],
              'var1735': [0.21303836196752177, 48],
              'var1736': [0.18004236290891976, 47],
              'var1737': [0.24330430689574017, 20],
              'var1738': [0.20960225935514237, 22],
              'var1739': [0.17735937867733584, 19],
              'var18': [0.2746528594963521, 3],
              'var1806': [0.2342668863261944, 47],
              'var1807': [0.20145916686279125, 47],
              'var1808': [0.16921628618498472, 44],
              'var1809': [0.2133678512591198, 16],
              'var1810': [0.18197222875970817, 16],
              'var1811': [0.1534008001882796, 14],
              'var1824': [0.19764650506001413, 34],
              'var1825': [0.17076959284537538, 38],
              'var1827': [0.15547187573546717, 14],
              'var19': [0.3884208048952695, 2],
              'var1968': [0.21219110378912687, 36],
              'var1969': [0.18573782066368558, 40],
              'var1970': [0.15589550482466463, 38],
              'var1971': [0.19166862791244998, 18],
              'var1972': [0.16709814073899742, 16],
              'var1977': [0.19990586020240056, 29],
              'var1978': [0.17171099082136973, 29],
              'var1980': [0.18945634266886327, 13],
              'var1981': [0.1624852906566251, 13],
              'var1986': [0.2275358907978348, 53],
              'var1987': [0.1970345963756178, 53],
              'var1988': [0.16314426923982114, 50],
              'var1989': [0.21275594257472347, 25],
              'var1990': [0.1853141915744881, 24],
              'var1991': [0.15029418686749824, 21],
              'var2': [1.0, 2],
              'var2004': [0.15062367615909625, 36],
              'var2031': [0.17519416333254884, 37],
              'var2032': [0.1507648858554954, 39],
              'var2244': [0.22419392798305485, 34],
              'var2245': [0.19402212285243586, 37],
              'var2246': [0.16295598964462227, 38],
              'var2247': [0.19745822546481526, 13],
              'var2248': [0.1716168510237703, 12],
              'var3': [0.9876206166156742, 32],
              'var4': [0.9876206166156742, 350],
              'var404': [0.19312779477524122, 44],
              'var405': [0.16450929630501293, 47],
              'var413': [0.16723935043539656, 24],
              'var431': [0.21317957166392093, 995],
              'var432': [0.17872440574252765, 843],
              'var434': [0.19406919275123558, 440],
              'var435': [0.1606024947046364, 370],
              'var437': [0.1726994586961638, 725],
              'var440': [0.24240997881854554, 233],
              'var441': [0.20908449046834549, 209],
              'var442': [0.1764650506001412, 186],
              'var443': [0.20169451635678984, 421],
              'var444': [0.1566486232054601, 383],
              'var445': [0.1705342433513768, 408],
              'var449': [0.23497293480819018, 106],
              'var450': [0.2008472581783949, 97],
              'var451': [0.16931042598258414, 89],
              'var452': [0.20353024240997883, 290],
              'var453': [0.15839020946104965, 123],
              'var454': [0.17204048011296777, 248],
              'var458': [0.20169451635678984, 3091],
              'var459': [0.1566486232054601, 1814],
              'var460': [0.1705342433513768, 2449],
              'var464': [0.1927041656860438, 196],
              'var465': [0.17011061426217933, 178],
              'var5': [1.0, 31],
              'var6': [0.9405507178159567, 335],
              'var654': [0.1686985172981878, 51],
              'var7': [1.0, 2],
              'var735': [0.19825841374441044, 65],
              'var736': [0.16347375853141916, 65],
              'var738': [0.15010590727229936, 289],
              'var744': [0.17627677100494235, 37],
              'var747': [0.15029418686749824, 148],
              'var753': [0.15010590727229936, 1827],
              'var759': [0.159331607437044, 56],
              'var789': [0.21317957166392093, 127],
              'var790': [0.17872440574252765, 124],
              'var792': [0.1755707225229466, 371],
              'var798': [0.19406919275123558, 59],
              'var799': [0.1606024947046364, 59],
              'var8': [1.0, 2],
              'var801': [0.17599435161214402, 179],
              'var803': [0.15062367615909625, 164],
              'var807': [0.1755707225229466, 2402],
              'var813': [0.1726994586961638, 117],
              'var843': [0.23572605318898565, 19],
              'var844': [0.2026829842315839, 20],
              'var845': [0.17034596375617791, 21],
              'var852': [0.2219345728406684, 10],
              'var853': [0.189832901859261, 10],
              'var854': [0.15980230642504117, 10],
              'var9': [1.0, 2],
              'var969': [0.23727935984937631, 37],
              'var970': [0.20390680160037655, 40],
              'var971': [0.17152271122617085, 39],
              'var978': [0.22471169686985173, 14],
              'var979': [0.19213932690044716, 15],
              'var980': [0.16135561308543186, 14]}

def change_timestamp(dt):

    timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
    timestamp = int(time.mktime(timeArray))
    return timestamp


def change_timestamp2(dt):
    timeArray = time.strptime(dt, "%Y-%m-%d")
    timestamp = int(time.mktime(timeArray))
    return timestamp


def get_feat_A(behavior, consumer, ccx, behavior_init_col=dealed_col):
    behavior['basedata_null_rate'] = behavior.apply(
        lambda x: len(x.ix[1:19][x.ix[1:19].isnull()])/18, 1)
    behavior['behavior_null_rate'] = behavior.apply(
    lambda x: len(x.ix[20:-1][x.ix[20:-1].isnull()])/2252, 1)

    
    train_behavior = behavior[list(
        dealed_col.keys())+['basedata_null_rate', 'behavior_null_rate']]
    train_behavior.drop("var19", 1, inplace=True)
    
    train_behavior.var3 = train_behavior.var3.fillna(-1)


    train_behavior.var3 = train_behavior.var3.apply(
        lambda x: int(str(x)[1:]) if x != -1 else x, 1)

    train_behavior.var4 = train_behavior.var4.fillna(-1)
    train_behavior.var4 = train_behavior.var4.apply(
        lambda x: int(str(x)[2:]) if x != -1 else x, 1)

    train_behavior.var5 = train_behavior.var5.fillna(-1)
    train_behavior.var5 = train_behavior.var5.apply(
        lambda x: int(str(x)[1:]) if x != -1 else x, 1)

    train_behavior.var6 = train_behavior.var6.fillna(-1)
    train_behavior.var6 = train_behavior.var6.apply(
        lambda x: int(str(x)[1:]) if x != -1 else x, 1)

    train_behavior.var11 = train_behavior.var11.fillna(-1)
    train_behavior.var11 = train_behavior.var11.apply(
        lambda x: int(str(x)[1:]) if x != -1 else x, 1)

    train_behavior.var12 = train_behavior.var12.fillna(-1)
    train_behavior.var12 = train_behavior.var12.apply(
        lambda x: int(str(x)[2:]) if x != -1 else x, 1)

    train_behavior.var13 = train_behavior.var13.fillna(-1)
    train_behavior.var13 = train_behavior.var13.apply(
        lambda x: int(str(x)[1:]) if x != -1 else x, 1)

    train_behavior.var18 = train_behavior.var18.fillna(-1)
    train_behavior.var18 = train_behavior.var18.apply(
        lambda x: int(str(x)[2:]) if x != -1 else x, 1)

    consumer_other = consumer.groupby(
        "ccx_id", as_index=False).first()[['ccx_id']]


    consumer['V_4_is_0'] = consumer['V_4'].apply(lambda x: 1 if x == 0 else 0, 1)
    consumer_other['V_4_is_0_rate'] = consumer.groupby("ccx_id", as_index=False)['V_4_is_0'].sum(
    )['V_4_is_0']/consumer.groupby("ccx_id", as_index=False)['V_4_is_0'].count()['V_4_is_0']

    consumer['V_9_is_0'] = consumer['V_9'].apply(lambda x: 1 if x == 0 else 0, 1)
    consumer_other['V_9_is_0_rate'] = consumer.groupby("ccx_id", as_index=False)['V_9_is_0'].sum(
    )['V_9_is_0']/consumer.groupby("ccx_id", as_index=False)['V_9_is_0'].count()['V_9_is_0']

    consumer['V_10_is_0'] = consumer['V_10'].apply(lambda x: 1 if x == 0 else 0, 1)
    consumer_other['V_10_is_0_rate'] = consumer.groupby("ccx_id", as_index=False)['V_10_is_0'].sum(
    )['V_10_is_0']/consumer.groupby("ccx_id", as_index=False)['V_10_is_0'].count()['V_10_is_0']

    consumer['is_v12*v13=v5'] = consumer.apply(
        lambda x: 1 if x['V_12']*x["V_13"] == x["V_5"] else 0, 1)

    consumer_other['is_v12*v13=v5_rate'] = consumer.groupby("ccx_id", as_index=False)['is_v12*v13=v5'].sum(
    )['is_v12*v13=v5']/consumer.groupby("ccx_id", as_index=False)['is_v12*v13=v5'].count()['is_v12*v13=v5']


    buy_many = consumer.groupby(['ccx_id', "V_7"], as_index=False).count()
    buy_many['is_buymany'] = buy_many['V_1'].apply(lambda x: 1 if x > 1 else 0, 1)
    consumer_other['is_buymany_rate'] = buy_many.groupby("ccx_id", as_index=False)['is_buymany'].sum(
    )['is_buymany']/buy_many.groupby("ccx_id", as_index=False)['is_buymany'].count()['is_buymany']
    consumer_other['buytime_count'] = buy_many.groupby("ccx_id", as_index=False)[
        'is_buymany'].count()['is_buymany']

    num_col=['V_4','V_5',"V_6",'V_9','V_10',"V_12",'V_13']
    num_data=None
    for i in num_col:
        temp=consumer.groupby("ccx_id",as_index=False)[i].agg({i+"_mean":"mean",i+"_max":"max",i+"_min":"min"
                                                        ,i+"_median":"median",i+"_sum":"sum",
                                                        i+"_std":"std",i+"_skew":"skew",i+"_last":"last",
                                                        })
        if i=='V_4':
            num_data=temp
        else:
            num_data=pd.merge(num_data,temp,on="ccx_id",how="left")
        print(i,"done")

    
    consumer.V_1=consumer.V_1.fillna(-1)
    consumer.V_1=consumer.V_1.apply(lambda x:int(str(x)[1:]) if x!=-1 else x,1)

    consumer.V_2=consumer.V_2.fillna(-1)
    consumer.V_2=consumer.V_2.apply(lambda x:int(str(x)[1:]) if x!=-1 else x,1)

    consumer.V_3=consumer.V_3.fillna(-1)
    consumer.V_3=consumer.V_3.apply(lambda x:int(str(x)[1:]) if x!=-1 else x,1)

    consumer.V_8=consumer.V_8.fillna(-1)
    consumer.V_8=consumer.V_8.apply(lambda x:int(str(x)[2:]) if x!=-1 else x,1)

    consumer.V_14=consumer.V_14.fillna(-1)
    consumer.V_14=consumer.V_14.apply(lambda x:int(str(x)[2:]) if x!=-1 else x,1)

    cate_col=['V_1','V_2','V_3','V_8','V_14']
    cate_data=None
    for i in cate_col:
        consumer[i]=consumer[i].fillna(-1)
        
        temp=consumer.groupby("ccx_id")[i].agg(lambda x: x.value_counts().index[0]).reset_index()
        
        if i=="V_1":
            cate_data=temp
        else:
            cate_data=pd.merge(cate_data,temp,on="ccx_id",how="left")
        print(i,'done')
    
    ccx.var_01=ccx.var_01.fillna(-1)
    ccx.var_01=ccx.var_01.apply(lambda x:int(str(x)[1:]) if x!=-1 else x,1)

    ccx.var_02=ccx.var_02.fillna(-1)
    ccx.var_02=ccx.var_02.apply(lambda x:int(str(x)[1:]) if x!=-1 else x,1)

    ccx.var_03=ccx.var_03.fillna(-1)
    ccx.var_03=ccx.var_03.apply(lambda x:int(str(x)[1:]) if x!=-1 else x,1)

    ccx.var_04=ccx.var_04.fillna(-1)
    ccx.var_04=ccx.var_04.apply(lambda x:int(str(x)[1:]) if x!=-1 else x,1)

    ccx.var_05=ccx.var_05.fillna(-1)
    ccx.var_05=ccx.var_05.apply(lambda x:int(str(x)[1:]) if x!=-1 else x,1)

    cate_col=['var_01','var_02','var_03','var_04','var_05']
    ccx_cate_data=None
    for i in cate_col:
        ccx[i]=ccx[i].fillna(-1)
        
        temp=ccx.groupby("ccx_id")[i].agg(lambda x: x.value_counts().index[0]).reset_index()
        
        if i=="var_01":
            ccx_cate_data=temp
        else:
            ccx_cate_data=pd.merge(ccx_cate_data,temp,on="ccx_id",how="left")
        print(i,'done')
    
    ###ccx
    ccx_count=ccx.groupby("ccx_id",as_index=False)['var_01'].count()
    ccx_count.rename(columns={"var_01":"ccx_count"}, inplace=True) 
    ccx_feature=pd.merge(ccx_cate_data,ccx_count,on="ccx_id",how="left")

    ####consumer
    consumer_feat=pd.merge(num_data,cate_data,on="ccx_id",how="left")
    consumer_feat=pd.merge(consumer_feat,consumer_other,on="ccx_id",how="left")

    ccx_time = ccx.groupby("ccx_id", as_index=False)['var_06'].last()
    consumer = pd.merge(consumer, ccx_time, on="ccx_id", how="left")
    consumer = consumer.sort_values(by=['ccx_id', "V_7"])
    consumer['hour'] = consumer['V_7'].apply(lambda x: int(x[11:13]), 1)
    consumer['weekday'] = consumer['V_7'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday(), 1)

    consumer.var_06 = consumer.var_06.fillna("1970-01-01")
    consumer.var_06 = consumer.var_06.apply(lambda x: change_timestamp2(x), 1)
    consumer = pd.merge(
        consumer, behavior[['ccx_id', 'var19']], on="ccx_id", how="left")

    consumer['var19'] = consumer['var19'].fillna("1970-01-01")
    consumer['var19'] = consumer['var19'].apply(lambda x: change_timestamp2(x), 1)

    consumer.V_11 = consumer.V_11.replace(
        '0000-00-00 00:00:00', "1970-01-01 00:00:00")
    consumer['V_7'] = consumer['V_7'].apply(lambda x: change_timestamp(x), 1)
    consumer['V_11'] = consumer['V_11'].apply(lambda x: change_timestamp(x), 1)
    consumer['consumer_V_7-V_11'] = consumer['V_7']-consumer['V_11']

    time_feat = consumer.groupby("ccx_id", as_index=False).last()


    time_dis = time_feat[['ccx_id']]

    time_dis['behaday-last_consumer'] = time_feat['var19']-time_feat['V_7']
    time_dis['behaday-ccx_time'] = time_feat['var19']-time_feat['var_06']

    time_dis['ccx_time'] = time_feat['var_06']
    time_V_7_V_11_feat = consumer.groupby("ccx_id", as_index=False)['consumer_V_7-V_11'].agg({'consumer_V_7-V_11_max': "max",
                                                                                            'consumer_V_7-V_11_min': "min", 'consumer_V_7-V_11_mean': "mean", 'consumer_V_7-V_11_median': "median",
                                                                                            'consumer_V_7-V_11_std': "std", 'consumer_V_7-V_11_skew': "skew"})
    time_V_7_feat = consumer.groupby("ccx_id", as_index=False)['V_7'].agg({'V_7_max': "max", 'V_7_min': "min",
                                                                        'V_7_mean': "mean", 'V_7_median': "median", 'V_7_std': "std", 'V_7_skew': "skew", 'V_7_last': "last"})
    time_dis = pd.merge(time_dis, time_V_7_V_11_feat, on="ccx_id", how="left")
    time_dis = pd.merge(time_dis, time_V_7_feat, on="ccx_id", how="left")
    time_dis['dur_day'] = time_dis['V_7_max']-time_dis['V_7_min']

    consumer['is_morning'] = consumer.hour.apply(
        lambda x: 1 if x > 6 & x < 11 else 0, 1)
    consumer['is_afternoon'] = consumer.hour.apply(
        lambda x: 1 if x >= 11 & x < 17 else 0, 1)
    consumer['is_evening'] = consumer.hour.apply(
        lambda x: 1 if x >= 17 & x <= 23 else 0, 1)
    consumer['is_night'] = consumer.hour.apply(
        lambda x: 1 if x == 24 & x <= 6 else 0, 1)

    time_feat['count'] = consumer.groupby("ccx_id", as_index=False).count()['V_2']

    time_dis['morning_order_count'] = consumer.groupby("ccx_id", as_index=False)[
        'is_morning'].sum()['is_morning']/time_feat['count']

    time_dis['afternoon_order_count'] = consumer.groupby("ccx_id", as_index=False)[
        'is_afternoon'].sum()['is_afternoon']/time_feat['count']

    time_dis['evening_order_count'] = consumer.groupby("ccx_id", as_index=False)[
        'is_evening'].sum()['is_evening']/time_feat['count']

    time_dis['night_order_count'] = consumer.groupby("ccx_id", as_index=False)[
        'is_night'].sum()['is_night']/time_feat['count']

    consumer['is_weekday'] = consumer.weekday.apply(lambda x: 1 if x < 5 else 0, 1)
    consumer['is_weekend'] = consumer.weekday.apply(
        lambda x: 1 if x >= 5 else 0, 1)

    time_dis['weekday_count'] = consumer.groupby("ccx_id", as_index=False)[
        'is_weekday'].sum()['is_weekday']/time_feat['count']
    time_dis['weekend_count'] = consumer.groupby("ccx_id", as_index=False)[
        'is_weekend'].sum()['is_weekend']/time_feat['count']
    time_dis['weekday_mode'] = consumer.groupby("ccx_id")['weekday'].agg(
        lambda x: x.value_counts().index[0]).reset_index()['weekday']

    data=pd.merge(train_behavior,consumer_feat,on="ccx_id",how="left")
    data=pd.merge(data,ccx_feature,on="ccx_id",how="left")
    data=pd.merge(data,time_dis,on="ccx_id",how="left")

    return data


def get_label(data, target):
    data = pd.merge(data, target, on="ccx_id", how="left")
    return data
def get_behaviorpred(behavior,target,train,test,test_behavior,flag='A'):
    behaviorpred = pd.merge(behavior, target, on="ccx_id", how="left")

    X = behaviorpred.iloc[:, 20:-3]
    if flag=='A':
        test_x = test_behavior.iloc[:,20:-2]
    else:
        test_x = test_behavior.iloc[:,19:-2]
    test_x=test_x.fillna(0)
    X = X.fillna(0)
    y = behaviorpred.target
    behaviorpred['behaviorpred'] = 1
    skf = StratifiedKFold(n_splits=5, random_state=1024)
    rlt_pred=0
    for train_index, test_index in skf.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clf = DecisionTreeClassifier(random_state=0, max_depth=12)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:, 1]
        behaviorpred.loc[test_index, 'behaviorpred'] = y_pred
        pred = clf.predict_proba(test_x)[:, 1]
        rlt_pred+=pred
    train = pd.merge(
        train, behaviorpred[["ccx_id", 'behaviorpred']], on="ccx_id", how="left")
    
    test['behaviorpred']=rlt_pred/5
    return train,test 
####交叉特征


def add(x, y):
    return x + y


def substract(x, y):
    return x - y


def times(x, y):
    return x * y


def divide(x, y):
    return (x + 0.001)/(y + 0.001)


CrossMethod = {
    #                 '+':add,
    #                '-':substract,
    '*': times,
    '/': divide, }


def get_cv_feat(cv_col, train):
    bh = pd.DataFrame()

    for i in range(len(cv_col)):
        for j in range(i+1, len(cv_col)):
            for k in CrossMethod:
                bh[cv_col[i]+k+cv_col[j]
                   ] = CrossMethod[k](train[cv_col[i]], train[cv_col[j]])
#                 print(cv_col[i]+k+cv_col[j],"done")
    return bh
print('train...')
data = get_feat_A(train_behavior_A, train_consumer_A, train_ccx_A)
data = get_label(data, target)

###cv_feat

raw_train = data
raw200 = ['var1', 'ccx_id', 'var17', 'V_13_skew', 'V_3',
          'basedata_null_rate', 'V_10_skew', 'ccx_count',
          'behaday-last_consumer', 'var6', 'V_13_max', 'V_2', 'var1312',
          'var1644', 'V_7_min', 'V_7_median', 'var13', 'V_7_skew', 'var440',
          'var4', 'consumer_V_7-V_11_median', 'V_10_median', 'var1310',
          'V_6_last', 'var11', 'var437', 'var1643', 'V_13_sum', 'V_14',
          'var1272', 'V_6_skew', 'V_10_last', 'V_13_median', 'var753',
          'V_7_max', 'V_7_std', 'V_5_min', 'V_5_last', 'V_6_min', 'V_7_mean',
          'var460', 'V_13_last', 'ccx_time', 'consumer_V_7-V_11_skew',
          'var441', 'V_13_std', 'is_buymany_rate', 'V_10_sum',
          'consumer_V_7-V_11_std', 'V_13_mean', 'var5', 'var1296',
          'V_5_skew', 'V_6_median', 'var1292', 'var1299', 'var431',
          'V_5_std', 'V_10_std', 'var155', 'V_12_std', 'var3', 'var_03',
          'var1311', 'dur_day', 'var443', 'V_4_skew', 'V_5_sum', 'V_6_mean',
          'weekday_count', 'var453', 'var803', 'var1273', 'V_10_max',
          'consumer_V_7-V_11_mean', 'V_12_max', 'consumer_V_7-V_11_max',
          'consumer_V_7-V_11_min', 'var_02', 'var1271', 'V_10_mean',
          'var1274', 'V_6_std', 'V_12_sum', 'is_v12*v13=v5_rate', 'var1282',
          'V_6_sum', 'var458', 'V_5_median', 'var1281', 'var1293',
          'night_order_count', 'V_12_skew', 'V_5_mean', 'var_04',
          'V_10_is_0_rate', 'V_6_max', 'var459', 'var_05', 'V_7_last',
          'var738', 'V_12_mean', 'V_13_min', 'var464', 'var1297', 'V_4_std',
          'V_1', 'V_10_min', 'var157', 'var449', 'var12', 'var434',
          'var1300', 'V_5_max', 'var1298', 'behaday-ccx_time', 'var452',
          'var442', 'var792', 'var1280', 'var1623', 'var435', 'var465',
          'var807', 'var1306', 'morning_order_count', 'var445', 'var798',
          'var1307', 'var1989', 'var405', 'var454', 'var1295', 'var1637',
          'var1305', 'var1646', 'var747', 'var444', 'var1166', 'var1308',
          'V_4_mean', 'var432', 'var735', 'var789', 'buytime_count',
          'var451', 'var1164', 'var1294', 'weekend_count', 'weekday_mode',
          'var801', 'var1990', 'V_4_sum', 'var736', 'var1286', 'var156',
          'var744', 'var1277', 'var1824', 'var1986', 'var1288', 'var1304',
          'var1622', 'V_8', 'var404', 'var450', 'var1978', 'var_01',
          'var1163', 'var1454', 'var1619', 'var1620', 'V_4_max', 'var759',
          'var1284', 'var1614', 'var1806', 'var18', 'var1167', 'var1309',
          'var1645', 'behavior_null_rate', 'var813', 'var1165', 'var1734',
          'var1738', 'var1970', 'var971', 'var1279', 'var1825', 'var1980',
          'var654', 'var790', 'var799', 'var843', 'var1169', 'var1278',
          'var1287', 'var1452', 'var1610', 'var1613']
fa30 = ['var1', 'ccx_id', 'var17', 'V_13_skew', 'V_3',
        'basedata_null_rate', 'V_10_skew', 'ccx_count',
        'behaday-last_consumer', 'var6', 'V_13_max', 'V_2', 'var1312',
        'var1644', 'V_7_min', 'V_7_median', 'var13', 'V_7_skew', 'var440',
        'var4', 'consumer_V_7-V_11_median', 'V_10_median', 'var1310',
        'V_6_last', 'var11', 'var437', 'var1643', 'V_13_sum', 'V_14',
        'var1272', 'V_6_skew']
cv_data1 = get_cv_feat(fa30, raw_train)
cv_feat1 = ['ccx_id*V_7_median', 'var1/V_7_min', 'var1*var437',
            'ccx_id/V_7_min', 'var17/var11', 'var1/V_7_median', 'var1*V_7_min',
            'ccx_id*ccx_count', 'var1*V_7_median', 'var440*var437',
            'basedata_null_rate*V_7_median', 'ccx_count/var4',
            'basedata_null_rate/V_7_min', 'var1644/V_7_min', 'var6/V_13_sum',
            'var1*basedata_null_rate', 'basedata_null_rate*V_7_min',
            'ccx_id/V_7_median', 'basedata_null_rate/V_7_median',
            'ccx_count/V_13_sum', 'var1644*var440', 'var1/V_13_skew',
            'var1/var440', 'V_3/ccx_count', 'var1/ccx_id', 'var1*var11',
            'var17*basedata_null_rate', 'basedata_null_rate/var440',
            'V_7_skew*V_10_median', 'basedata_null_rate/var11',
            'V_7_median/var11', 'V_13_skew/V_7_skew',
            'behaday-last_consumer*V_13_max', 'var1/var13', 'ccx_id/V_13_skew',
            'V_3*var6', 'V_3/V_7_skew', 'basedata_null_rate*ccx_count',
            'ccx_count/V_13_max', 'var6/var4', 'var1/basedata_null_rate',
            'basedata_null_rate*var13', 'V_2/var1644', 'ccx_id/ccx_count',
            'V_3/V_13_sum', 'V_10_median*V_13_sum', 'var11*V_14',
            'ccx_id*V_7_min', 'V_3*V_10_skew', 'V_3/var1643',
            'basedata_null_rate/V_6_last', 'behaday-last_consumer*var1643',
            'var6/var11', 'V_7_skew*V_14', 'basedata_null_rate/var1310',
            'behaday-last_consumer/V_7_skew',
            'behaday-last_consumer/consumer_V_7-V_11_median',
            'var1644/V_6_last', 'var440*var1643',
            'consumer_V_7-V_11_median/var1272', 'var1/var1644',
            'V_10_skew/var1643', 'basedata_null_rate*V_13_max',
            'basedata_null_rate/V_2', 'ccx_count/var13', 'V_13_max/var4',
            'V_13_max*var11', 'V_13_max*V_13_sum', 'var1*var13',
            'var17/V_7_median', 'V_3*V_13_sum', 'behaday-last_consumer/V_14',
            'var6/V_2', 'V_7_min*V_7_median', 'V_7_min/V_7_median',
            'ccx_id/var11', 'var17*V_13_skew', 'V_13_skew*V_10_skew',
            'V_3*ccx_count', 'V_3/V_13_max', 'ccx_count/behaday-last_consumer',
            'var1312*var13', 'var13/V_13_sum', 'var1*consumer_V_7-V_11_median',
            'ccx_id/basedata_null_rate', 'ccx_id/V_6_last', 'ccx_count/V_14',
            'V_2/var1643', 'var1312*var1643', 'V_7_skew*V_6_skew',
            'var4*V_10_median', 'var1/var437', 'ccx_id/V_10_skew',
            'ccx_id/V_13_sum', 'var17*behaday-last_consumer', 'V_3/var6',
            'V_3/V_7_min', 'V_10_skew/V_7_skew', 'V_2/V_7_skew',
            'V_7_min/var1643', 'V_10_median/V_13_sum']
temp = raw_train[raw200].join(cv_data1[cv_feat1])
sa30 = ['var1/V_7_min', 'var1*var437', 'ccx_id', 'var1/V_7_median',
        'ccx_id*ccx_count', 'behaday-last_consumer*var1643',
        'var1644/V_7_min', 'ccx_id*V_7_median', 'ccx_count/var4',
        'var1*var11', 'var17/var11', 'var6/var4', 'var1*V_7_median',
        'basedata_null_rate*V_7_median', 'ccx_id/V_7_min', 'var6/V_13_sum',
        'var1/V_13_skew', 'var1/ccx_id', 'var1*V_7_min',
        'var4*V_10_median', 'var1*basedata_null_rate', 'var1/var1644',
        'basedata_null_rate/V_2', 'V_13_max*var11',
        'ccx_count/behaday-last_consumer', 'basedata_null_rate*V_7_min',
        'basedata_null_rate/V_7_median', 'ccx_id/var11', 'V_3*var6',
        'var17*V_13_skew', 'var1644*var440']
cv_data2 = get_cv_feat(sa30, temp)
cv_feat2 = ['var1*var11/var1*basedata_null_rate', 'ccx_id/ccx_id/V_7_min',
            'var1/V_7_min*var1/V_7_median',
            'var17/var11*basedata_null_rate/V_7_median',
            'ccx_id*ccx_count/var1/ccx_id',
            'var1*var11/ccx_count/behaday-last_consumer',
            'var6/V_13_sum*var1/V_13_skew', 'var1*var437*var1644*var440',
            'var4*V_10_median/basedata_null_rate/V_2',
            'var1*basedata_null_rate/ccx_count/behaday-last_consumer',
            'var1/V_7_min/var1/V_7_median', 'var1/V_7_median*var1644/V_7_min',
            'V_13_max*var11*basedata_null_rate/V_7_median',
            'var1644/V_7_min/basedata_null_rate/V_7_median',
            'var6/V_13_sum/V_13_max*var11',
            'V_13_max*var11*basedata_null_rate*V_7_min',
            'var1*var437/var4*V_10_median',
            'basedata_null_rate*V_7_median*basedata_null_rate*V_7_min',
            'var17/var11/var1*V_7_median', 'var6/V_13_sum/var4*V_10_median',
            'var1/V_7_min/var1*var437', 'ccx_id/var1*var11',
            'var1644/V_7_min*ccx_count/behaday-last_consumer',
            'ccx_count/var4*basedata_null_rate*V_7_median',
            'var1*var11/basedata_null_rate*V_7_median',
            'var6/var4*var1/V_13_skew', 'var1*V_7_median/var1/V_13_skew',
            'var1*var437/var1*V_7_min', 'ccx_id/var1/V_7_median',
            'ccx_id/basedata_null_rate/V_7_median',
            'var1/V_7_median/var1*var11', 'var1/V_7_median*var1*V_7_median',
            'ccx_id*V_7_median*ccx_id/V_7_min', 'var1/V_7_min*var1644/V_7_min',
            'var1/V_7_min*var17*V_13_skew', 'var1*var437/var1644*var440',
            'ccx_id*ccx_count/var4', 'ccx_id/var17/var11',
            'ccx_id*ccx_count/var17/var11',
            'behaday-last_consumer*var1643*var1644*var440',
            'var1*var11/basedata_null_rate*V_7_min',
            'var4*V_10_median*basedata_null_rate/V_2',
            'var1*var437*var1/ccx_id', 'ccx_id*V_13_max*var11',
            'var1/V_7_median*ccx_count/var4', 'var1/V_7_median/var17/var11',
            'var1/V_7_median*var6/V_13_sum', 'var1/V_7_median*var17*V_13_skew',
            'behaday-last_consumer*var1643*var1*basedata_null_rate',
            'var1644/V_7_min/var17/var11', 'var1*var11/var17/var11',
            'var6/var4/V_13_max*var11', 'var4*V_10_median/V_13_max*var11',
            'var1*var437/var1*V_7_median', 'ccx_id*ccx_id*ccx_count',
            'var1644/V_7_min*var1*V_7_median',
            'var1644/V_7_min/var1/V_13_skew', 'var17/var11*V_3*var6',
            'basedata_null_rate*V_7_median/var1*V_7_min',
            'basedata_null_rate*V_7_median*var1/var1644',
            'basedata_null_rate*V_7_median/basedata_null_rate*V_7_min',
            'var1/V_13_skew/basedata_null_rate/V_2',
            'var1/V_13_skew*V_13_max*var11',
            'var1*V_7_min/basedata_null_rate*V_7_min',
            'var4*V_10_median*V_13_max*var11',
            'var4*V_10_median/basedata_null_rate/V_7_median',
            'var1*basedata_null_rate*V_3*var6',
            'basedata_null_rate/V_2/basedata_null_rate*V_7_min',
            'ccx_id/var11*var1644*var440', 'var1/V_7_min/var17/var11',
            'var1/V_7_min/var1644*var440', 'var1*var437/var17/var11',
            'var1/V_7_median*basedata_null_rate/V_7_median',
            'ccx_id*ccx_count*basedata_null_rate*V_7_median',
            'ccx_id*ccx_count/V_13_max*var11',
            'ccx_id*ccx_count*basedata_null_rate*V_7_min',
            'behaday-last_consumer*var1643/basedata_null_rate*V_7_median',
            'var1644/V_7_min/var1*V_7_min',
            'var1644/V_7_min*basedata_null_rate/V_2',
            'ccx_id*V_7_median*var6/V_13_sum', 'var1*var11*var6/V_13_sum',
            'var6/var4*V_13_max*var11', 'var6/var4*var1644*var440',
            'ccx_id/V_7_min/basedata_null_rate/V_7_median',
            'var6/V_13_sum/ccx_count/behaday-last_consumer',
            'var1/ccx_id*V_3*var6', 'var1*V_7_min/basedata_null_rate/V_2',
            'var1*basedata_null_rate/basedata_null_rate/V_2',
            'basedata_null_rate/V_2/V_13_max*var11',
            'basedata_null_rate/V_2/basedata_null_rate/V_7_median',
            'basedata_null_rate/V_2*var1644*var440',
            'V_13_max*var11/ccx_id/var11', 'var1/V_7_min/var1*V_7_median',
            'var1/V_7_min/basedata_null_rate/V_2',
            'var1/V_7_median*var1*V_7_min',
            'ccx_id*ccx_count*ccx_id*V_7_median',
            'ccx_id*ccx_count/ccx_id/V_7_min',
            'ccx_id*ccx_count/var4*V_10_median',
            'behaday-last_consumer*var1643*var1644/V_7_min',
            'behaday-last_consumer*var1643*ccx_id/var11',
            'var1644/V_7_min*var1/ccx_id']

train = raw_train[raw200].join(cv_data1[cv_feat1])
train = train.join(cv_data2[cv_feat2])
del cv_data1,cv_data2
train['target'] = raw_train['target']
label = 'target'
features = ['var1*var11/var1*basedata_null_rate',
            'behaday-last_consumer/V_7_skew', 'ccx_id/ccx_id/V_7_min',
            'var17/var11*basedata_null_rate/V_7_median', 'ccx_id/var1*var11',
            'var1/V_7_min*var1/V_7_median',
            'var1*basedata_null_rate/ccx_count/behaday-last_consumer',
            'var1*var11/basedata_null_rate*V_7_median', 'var440*var437',
            'var1*consumer_V_7-V_11_median',
            'var1644/V_7_min/basedata_null_rate/V_7_median',
            'var17/var11/var1*V_7_median', 'var1644/V_7_min',
            'behaday-last_consumer*var1643/basedata_null_rate*V_7_median',
            'var1*var437', 'V_13_max*var11*basedata_null_rate/V_7_median',
            'var1/V_7_median/var1*var11', 'V_3/V_7_skew', 'var1312*var1643',
            'var6/V_13_sum/V_13_max*var11', 'var1/V_7_min/var1*var437',
            'ccx_count/var4', 'basedata_null_rate/var440',
            'ccx_id*ccx_count/var1/ccx_id',
            'var1644/V_7_min*ccx_count/behaday-last_consumer',
            'ccx_id*ccx_count/var4',
            'basedata_null_rate*V_7_median*var1/var1644',
            'var4*V_10_median/basedata_null_rate/V_7_median', 'V_13_skew',
            'V_12_sum', 'behaday-last_consumer*V_13_max',
            'behaday-last_consumer*var1643*ccx_id/var11', 'V_4_skew',
            'var1/V_7_min', 'basedata_null_rate*V_7_min', 'var1/var440',
            'V_7_min*V_7_median',
            'behaday-last_consumer*var1643*var1*basedata_null_rate',
            'var1*var11/var17/var11', 'ccx_id*ccx_id*ccx_count',
            'var1/V_13_skew*V_13_max*var11', 'var1644/V_7_min/var1*V_7_min',
            'var1272', 'V_13_skew/V_7_skew', 'V_3*var6',
            'ccx_count/var4*basedata_null_rate*V_7_median',
            'var1*V_7_median/var1/V_13_skew',
            'var1/V_7_median*var1*V_7_median',
            'var1/V_7_median*ccx_count/var4', 'night_order_count',
            'morning_order_count', 'ccx_count/behaday-last_consumer',
            'var6/V_13_sum*var1/V_13_skew',
            'V_13_max*var11*basedata_null_rate*V_7_min',
            'var1*var437/var4*V_10_median', 'ccx_id/var1/V_7_median',
            'ccx_id/var17/var11', 'var1*var11/basedata_null_rate*V_7_min',
            'var1/V_7_median*var6/V_13_sum', 'var1*var11*var6/V_13_sum',
            'var1*basedata_null_rate/basedata_null_rate/V_2',
            'behaday-last_consumer*var1643*var1644/V_7_min',
            'behaday-last_consumer', 'var792', 'V_13_max/var4',
            'V_13_max*V_13_sum', 'V_13_skew*V_10_skew', 'ccx_id/V_10_skew',
            'var1*var11/ccx_count/behaday-last_consumer',
            'var4*V_10_median/basedata_null_rate/V_2',
            'var1/V_7_median*var1644/V_7_min',
            'basedata_null_rate*V_7_median*basedata_null_rate*V_7_min',
            'ccx_id/basedata_null_rate/V_7_median',
            'var4*V_10_median*basedata_null_rate/V_2',
            'var1/V_7_min/var1644*var440', 'ccx_id*V_7_median*var6/V_13_sum',
            'var6/var4*var1644*var440', 'var440', 'var753', 'var738',
            'V_3/ccx_count', 'var1/ccx_id', 'var6/var4', 'V_2/var1644',
            'var11*V_14', 'behaday-last_consumer/consumer_V_7-V_11_median',
            'consumer_V_7-V_11_median/var1272', 'V_3/V_7_min',
            'var1/V_7_median/var17/var11',
            'var1/V_13_skew/basedata_null_rate/V_2',
            'var4*V_10_median*V_13_max*var11',
            'var1*basedata_null_rate*V_3*var6',
            'var6/V_13_sum/ccx_count/behaday-last_consumer', 'V_10_sum',
            'var1644*var440', 'V_10_median*V_13_sum', 'var1/var1644',
            'ccx_count/V_14', 'ccx_id*V_7_median*ccx_id/V_7_min',
            'ccx_id*ccx_count/var17/var11',
            'var1/V_7_min/basedata_null_rate/V_2', 'V_3', 'V_13_max',
            'consumer_V_7-V_11_median', 'V_13_last', 'V_6_mean', 'var458',
            'weekday_mode', 'var1/V_7_median', 'V_7_skew*V_10_median',
            'V_3/V_13_sum', 'var440*var1643']
print('test...')
test_A = get_feat_A(test_behavior_A, test_consumer_A, test_ccx_A)
###cv_feat test
cv_test1 = get_cv_feat(fa30, test_A)
temp = test_A[raw200].join(cv_test1[cv_feat1])
cv_test2 = get_cv_feat(sa30, temp)
test_A = test_A[raw200].join(cv_test1[cv_feat1])
test_A = test_A.join(cv_test2[cv_feat2])
del cv_test1, cv_test2

# print('behaviorpred...')
# train,test_A=get_behaviorpred(train_behavior_A,target,train,test_A,test_behavior_A)
print("train-shape:",train.shape)
print("test-shape:", test_A.shape)

print("train...")

params = {
    'boosting_type': 'gbdt',
    'metric': 'auc',
    #     "task":"train",
    'is_unbalance': 'True',
    'learning_rate': 0.01,
    'verbose': 0,
    'num_leaves': 32,
    #             'max_depth':3,
    #             'max_bin':10,
    #             'lambda_l2': 10,
    'objective': 'binary',
    'feature_fraction': 0.4,
    'bagging_fraction': 0.7,  # 0.9是目前最优的
    #             'bagging_freq':1,  # 3是目前最优的
    #             'min_data': 500,
    'seed': 1024,
    'nthread': 12,
    #             'silent': True,
}
dtrain = lgb.Dataset(train[features], train[label])
model = lgb.train(params, dtrain, num_boost_round=400,
                  verbose_eval=200,
                  )
print("predict...")
pred = model.predict(test_A[features])

test_A['prob'] = pred
predict_result_A = test_A[['ccx_id', 'prob']]
predict_result_A.to_csv('./predict_result_A.csv',
                        encoding='utf-8', index=False)

del test_A,train
def get_feat_B(behavior, consumer, behavior_init_col=dealed_col):
    behavior['basedata_null_rate'] = behavior.apply(
        lambda x: len(x.ix[1:19][x.ix[1:19].isnull()])/18, 1)
    behavior['behavior_null_rate'] = behavior.apply(
        lambda x: len(x.ix[20:-1][x.ix[20:-1].isnull()])/2252, 1)

    train_behavior = behavior[list(
        dealed_col.keys())+['basedata_null_rate', 'behavior_null_rate']]
    train_behavior.drop("var19", 1, inplace=True)

    train_behavior.var3 = train_behavior.var3.fillna(-1)

    train_behavior.var3 = train_behavior.var3.apply(
        lambda x: int(str(x)[1:]) if x != -1 else x, 1)

    train_behavior.var4 = train_behavior.var4.fillna(-1)
    train_behavior.var4 = train_behavior.var4.apply(
        lambda x: int(str(x)[2:]) if x != -1 else x, 1)

    train_behavior.var5 = train_behavior.var5.fillna(-1)
    train_behavior.var5 = train_behavior.var5.apply(
        lambda x: int(str(x)[1:]) if x != -1 else x, 1)

    train_behavior.var6 = train_behavior.var6.fillna(-1)
    train_behavior.var6 = train_behavior.var6.apply(
        lambda x: int(str(x)[1:]) if x != -1 else x, 1)

    train_behavior.var11 = train_behavior.var11.fillna(-1)
    train_behavior.var11 = train_behavior.var11.apply(
        lambda x: int(str(x)[1:]) if x != -1 else x, 1)

    train_behavior.var12 = train_behavior.var12.fillna(-1)
    train_behavior.var12 = train_behavior.var12.apply(
        lambda x: int(str(x)[2:]) if x != -1 else x, 1)

    train_behavior.var13 = train_behavior.var13.fillna(-1)
    train_behavior.var13 = train_behavior.var13.apply(
        lambda x: int(str(x)[1:]) if x != -1 else x, 1)

    train_behavior.var18 = train_behavior.var18.fillna(-1)
    train_behavior.var18 = train_behavior.var18.apply(
        lambda x: int(str(x)[2:]) if x != -1 else x, 1)

    consumer_other = consumer.groupby(
        "ccx_id", as_index=False).first()[['ccx_id']]

    consumer['V_4_is_0'] = consumer['V_4'].apply(
        lambda x: 1 if x == 0 else 0, 1)
    consumer_other['V_4_is_0_rate'] = consumer.groupby("ccx_id", as_index=False)['V_4_is_0'].sum(
    )['V_4_is_0']/consumer.groupby("ccx_id", as_index=False)['V_4_is_0'].count()['V_4_is_0']

    consumer['V_9_is_0'] = consumer['V_9'].apply(
        lambda x: 1 if x == 0 else 0, 1)
    consumer_other['V_9_is_0_rate'] = consumer.groupby("ccx_id", as_index=False)['V_9_is_0'].sum(
    )['V_9_is_0']/consumer.groupby("ccx_id", as_index=False)['V_9_is_0'].count()['V_9_is_0']

    consumer['V_10_is_0'] = consumer['V_10'].apply(
        lambda x: 1 if x == 0 else 0, 1)
    consumer_other['V_10_is_0_rate'] = consumer.groupby("ccx_id", as_index=False)['V_10_is_0'].sum(
    )['V_10_is_0']/consumer.groupby("ccx_id", as_index=False)['V_10_is_0'].count()['V_10_is_0']

    consumer['is_v12*v13=v5'] = consumer.apply(
        lambda x: 1 if x['V_12']*x["V_13"] == x["V_5"] else 0, 1)

    consumer_other['is_v12*v13=v5_rate'] = consumer.groupby("ccx_id", as_index=False)['is_v12*v13=v5'].sum(
    )['is_v12*v13=v5']/consumer.groupby("ccx_id", as_index=False)['is_v12*v13=v5'].count()['is_v12*v13=v5']

    buy_many = consumer.groupby(['ccx_id', "V_7"], as_index=False).count()
    buy_many['is_buymany'] = buy_many['V_1'].apply(
        lambda x: 1 if x > 1 else 0, 1)
    consumer_other['is_buymany_rate'] = buy_many.groupby("ccx_id", as_index=False)['is_buymany'].sum(
    )['is_buymany']/buy_many.groupby("ccx_id", as_index=False)['is_buymany'].count()['is_buymany']
    consumer_other['buytime_count'] = buy_many.groupby("ccx_id", as_index=False)[
        'is_buymany'].count()['is_buymany']

    num_col = ['V_4', 'V_5', "V_6", 'V_9', 'V_10', "V_12", 'V_13']
    num_data = None
    for i in num_col:
        temp = consumer.groupby("ccx_id", as_index=False)[i].agg({i+"_mean": "mean", i+"_max": "max", i+"_min": "min", i+"_median": "median", i+"_sum": "sum",
                                                                  i+"_std": "std", i+"_skew": "skew", i+"_last": "last",
                                                                  })
        if i == 'V_4':
            num_data = temp
        else:
            num_data = pd.merge(num_data, temp, on="ccx_id", how="left")
        print(i, "done")

    consumer.V_1 = consumer.V_1.fillna(-1)
    consumer.V_1 = consumer.V_1.apply(
        lambda x: int(str(x)[1:]) if x != -1 else x, 1)

    consumer.V_2 = consumer.V_2.fillna(-1)
    consumer.V_2 = consumer.V_2.apply(
        lambda x: int(str(x)[1:]) if x != -1 else x, 1)

    consumer.V_3 = consumer.V_3.fillna(-1)
    consumer.V_3 = consumer.V_3.apply(
        lambda x: int(str(x)[1:]) if x != -1 else x, 1)

    consumer.V_8 = consumer.V_8.fillna(-1)
    consumer.V_8 = consumer.V_8.apply(
        lambda x: int(str(x)[2:]) if x != -1 else x, 1)

    consumer.V_14 = consumer.V_14.fillna(-1)
    consumer.V_14 = consumer.V_14.apply(
        lambda x: int(str(x)[2:]) if x != -1 else x, 1)

    cate_col = ['V_1', 'V_2', 'V_3', 'V_8', 'V_14']
    cate_data = None
    for i in cate_col:
        consumer[i] = consumer[i].fillna(-1)

        temp = consumer.groupby("ccx_id")[i].agg(
            lambda x: x.value_counts().index[0]).reset_index()

        if i == "V_1":
            cate_data = temp
        else:
            cate_data = pd.merge(cate_data, temp, on="ccx_id", how="left")
        print(i, 'done')

    

    
    ####consumer
    consumer_feat = pd.merge(num_data, cate_data, on="ccx_id", how="left")
    consumer_feat = pd.merge(
        consumer_feat, consumer_other, on="ccx_id", how="left")

    consumer = consumer.sort_values(by=['ccx_id', "V_7"])
    consumer['hour'] = consumer['V_7'].apply(lambda x: int(x[11:13]), 1)
    consumer['weekday'] = consumer['V_7'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday(), 1)

    consumer = pd.merge(
        consumer, behavior[['ccx_id', 'var19']], on="ccx_id", how="left")

    consumer['var19'] = consumer['var19'].fillna("1970-01-01")
    consumer['var19'] = consumer['var19'].apply(
        lambda x: change_timestamp2(x), 1)

    consumer.V_11 = consumer.V_11.replace(
        '0000-00-00 00:00:00', "1970-01-01 00:00:00")
    consumer['V_7'] = consumer['V_7'].apply(lambda x: change_timestamp(x), 1)
    consumer['V_11'] = consumer['V_11'].apply(lambda x: change_timestamp(x), 1)
    consumer['consumer_V_7-V_11'] = consumer['V_7']-consumer['V_11']

    time_feat = consumer.groupby("ccx_id", as_index=False).last()

    time_dis = time_feat[['ccx_id']]

    time_dis['behaday-last_consumer'] = time_feat['var19']-time_feat['V_7']
    time_V_7_V_11_feat = consumer.groupby("ccx_id", as_index=False)['consumer_V_7-V_11'].agg({'consumer_V_7-V_11_max': "max",
                                                                                              'consumer_V_7-V_11_min': "min", 'consumer_V_7-V_11_mean': "mean", 'consumer_V_7-V_11_median': "median",
                                                                                              'consumer_V_7-V_11_std': "std", 'consumer_V_7-V_11_skew': "skew"})
    time_V_7_feat = consumer.groupby("ccx_id", as_index=False)['V_7'].agg({'V_7_max': "max", 'V_7_min': "min",
                                                                           'V_7_mean': "mean", 'V_7_median': "median", 'V_7_std': "std", 'V_7_skew': "skew", 'V_7_last': "last"})
    time_dis = pd.merge(time_dis, time_V_7_V_11_feat, on="ccx_id", how="left")
    time_dis = pd.merge(time_dis, time_V_7_feat, on="ccx_id", how="left")
    time_dis['dur_day'] = time_dis['V_7_max']-time_dis['V_7_min']

    consumer['is_morning'] = consumer.hour.apply(
        lambda x: 1 if x > 6 & x < 11 else 0, 1)
    consumer['is_afternoon'] = consumer.hour.apply(
        lambda x: 1 if x >= 11 & x < 17 else 0, 1)
    consumer['is_evening'] = consumer.hour.apply(
        lambda x: 1 if x >= 17 & x <= 23 else 0, 1)
    consumer['is_night'] = consumer.hour.apply(
        lambda x: 1 if x == 24 & x <= 6 else 0, 1)

    time_feat['count'] = consumer.groupby(
        "ccx_id", as_index=False).count()['V_2']

    time_dis['morning_order_count'] = consumer.groupby("ccx_id", as_index=False)[
        'is_morning'].sum()['is_morning']/time_feat['count']

    time_dis['afternoon_order_count'] = consumer.groupby("ccx_id", as_index=False)[
        'is_afternoon'].sum()['is_afternoon']/time_feat['count']

    time_dis['evening_order_count'] = consumer.groupby("ccx_id", as_index=False)[
        'is_evening'].sum()['is_evening']/time_feat['count']

    time_dis['night_order_count'] = consumer.groupby("ccx_id", as_index=False)[
        'is_night'].sum()['is_night']/time_feat['count']

    consumer['is_weekday'] = consumer.weekday.apply(
        lambda x: 1 if x < 5 else 0, 1)
    consumer['is_weekend'] = consumer.weekday.apply(
        lambda x: 1 if x >= 5 else 0, 1)

    time_dis['weekday_count'] = consumer.groupby("ccx_id", as_index=False)[
        'is_weekday'].sum()['is_weekday']/time_feat['count']
    time_dis['weekend_count'] = consumer.groupby("ccx_id", as_index=False)[
        'is_weekend'].sum()['is_weekend']/time_feat['count']
    time_dis['weekday_mode'] = consumer.groupby("ccx_id")['weekday'].agg(
        lambda x: x.value_counts().index[0]).reset_index()['weekday']

    data = pd.merge(train_behavior, consumer_feat, on="ccx_id", how="left")
    data = pd.merge(data, time_dis, on="ccx_id", how="left")

   

    return data

print("taskB.....")
print("train....")
print("data process....")
data.drop(['var_01', 'var_02', 'var_03', "var_04", 'var_05',
           "ccx_count", 'ccx_time', 'behaday-ccx_time'], 1, inplace=True)
###cv_feat  train
raw_train = data
raw200 = ['var1', 'ccx_id', 'V_3', 'var17', 'basedata_null_rate', 'V_5_last',
          'V_2', 'consumer_V_7-V_11_median', 'V_13_median', 'var4',
          'V_7_skew', 'var13', 'V_10_skew', 'V_13_std', 'var1644',
          'V_7_mean', 'var1310', 'V_14', 'V_10_std', 'V_13_skew', 'V_7_min',
          'behaday-last_consumer', 'V_13_sum', 'V_7_std', 'V_13_max',
          'var440', 'var6', 'var1272', 'V_6_last', 'V_13_last', 'var1296',
          'var1312', 'consumer_V_7-V_11_skew', 'var1311', 'var3', 'var437',
          'var753', 'V_6_skew', 'V_7_max', 'V_5_std', 'V_6_min', 'var1643',
          'var1292', 'V_5_skew', 'var460', 'dur_day', 'var1274',
          'V_10_median', 'V_4_skew', 'V_13_mean', 'consumer_V_7-V_11_max',
          'consumer_V_7-V_11_std', 'var431', 'var803', 'is_v12*v13=v5_rate',
          'var155', 'V_5_min', 'var11', 'V_6_mean', 'V_10_max', 'V_7_median',
          'var458', 'V_5_median', 'var1273', 'var441', 'V_6_std', 'V_6_max',
          'V_10_sum', 'is_buymany_rate', 'var1299', 'V_6_sum', 'V_12_mean',
          'var449', 'var1295', 'V_1', 'var5', 'V_10_mean', 'V_13_min',
          'V_12_skew', 'consumer_V_7-V_11_min', 'var1282', 'V_5_sum',
          'V_12_std', 'V_5_max', 'V_7_last', 'V_5_mean', 'V_12_max',
          'var434', 'var1271', 'V_6_median', 'V_12_sum', 'var1297',
          'weekday_count', 'var1294', 'var1298', 'var1306', 'var1308',
          'var405', 'var738', 'var1280', 'var442', 'var443', 'var453',
          'var1293', 'V_4_mean', 'V_10_is_0_rate', 'morning_order_count',
          'var12', 'V_10_min', 'var1281', 'consumer_V_7-V_11_mean', 'var464',
          'var1305', 'V_10_last', 'night_order_count', 'var435', 'var1646',
          'var1989', 'var445', 'var792', 'var1623', 'var1637', 'V_4_std',
          'var156', 'var1166', 'var1300', 'var432', 'var465', 'var747',
          'var450', 'var452', 'var454', 'var1286', 'var459', 'weekend_count',
          'var801', 'var807', 'var451', 'var1620', 'var736', 'var1307',
          'var789', 'var1622', 'buytime_count', 'var157', 'var444', 'var813',
          'var1164', 'var1645', 'var1986', 'weekday_mode', 'var1167',
          'var1304', 'var798', 'var1619', 'var1968', 'var1977', 'var1309',
          'var1987', 'var744', 'var799', 'var1163', 'var1825', 'var971',
          'var1169', 'var1990', 'var790', 'var970', 'var1165', 'var1278',
          'var1621', 'var1738', 'var1824', 'var1988', 'behavior_null_rate',
          'var843', 'var1613', 'var1614', 'V_4_sum', 'var735', 'var1980',
          'var1275', 'var1277', 'var1288', 'var1737', 'var1971', 'var759',
          'var1970', 'var1978', 'V_8', 'V_4_is_0_rate', 'var1287', 'var1452',
          'var1639', 'var1734', 'var1827', 'var1291', 'var1624', 'var1641',
          'var969', 'var1279']
fa30 = ['var1', 'ccx_id', 'V_3', 'var17', 'basedata_null_rate', 'V_5_last',
        'V_2', 'consumer_V_7-V_11_median', 'V_13_median', 'var4',
        'V_7_skew', 'var13', 'V_10_skew', 'V_13_std', 'var1644',
        'V_7_mean', 'var1310', 'V_14', 'V_10_std', 'V_13_skew', 'V_7_min',
        'behaday-last_consumer', 'V_13_sum', 'V_7_std', 'V_13_max',
        'var440', 'var6', 'var1272', 'V_6_last', 'V_13_last', 'var1296']
cv_data1 = get_cv_feat(fa30, raw_train)
cv_feat1 = ['var1/V_7_mean', 'var1/V_7_min', 'var1*basedata_null_rate',
            'ccx_id*V_7_mean', 'var1*V_7_mean', 'var17*basedata_null_rate',
            'ccx_id/V_7_mean', 'basedata_null_rate*V_7_mean',
            'var1644/V_7_min', 'var1/var13', 'basedata_null_rate/V_7_min',
            'ccx_id/V_7_min', 'V_13_median/var4', 'var1644*var440',
            'var1*V_7_min', 'V_2/var1644', 'var1*var13', 'var1/var440',
            'basedata_null_rate*V_7_min', 'V_7_skew/V_13_skew', 'var1/ccx_id',
            'V_3/var1644', 'basedata_null_rate/V_7_mean', 'V_3*V_13_sum',
            'basedata_null_rate/var1310', 'V_14/V_6_last', 'V_3/V_13_sum',
            'V_3*var6', 'V_2*V_13_median', 'V_13_std*V_13_last',
            'V_7_mean*var440', 'var1/var1644', 'V_14/V_7_min', 'V_3/V_13_last',
            'var1/var17', 'V_5_last/var1644', 'consumer_V_7-V_11_median/V_14',
            'var4/V_13_max', 'V_13_skew*behaday-last_consumer', 'var1*ccx_id',
            'var1/basedata_null_rate', 'var1/V_13_skew', 'V_3/V_6_last',
            'var13*V_14', 'var13/V_13_sum', 'V_13_std/V_13_sum',
            'var1*V_13_skew', 'ccx_id*basedata_null_rate', 'ccx_id/var1272',
            'V_3/V_13_skew', 'var17/basedata_null_rate',
            'basedata_null_rate*var13', 'V_13_median*V_13_sum',
            'V_7_mean/V_14', 'var1/V_7_skew', 'ccx_id*V_7_min', 'V_3/V_7_std',
            'var17*V_13_max', 'V_2*V_13_last',
            'consumer_V_7-V_11_median/V_6_last', 'var4*V_13_std',
            'V_10_skew*V_13_std', 'var1644/V_6_last',
            'V_14*behaday-last_consumer', 'V_10_std/V_13_sum',
            'V_7_std*V_6_last', 'V_3*V_13_median', 'V_3*V_13_skew', 'V_3/var6',
            'basedata_null_rate/var440', 'V_2/var4', 'V_2/var440',
            'V_13_median*var4', 'V_13_median/V_7_std', 'V_13_median/var1272',
            'var4/var6', 'V_7_mean*behaday-last_consumer', 'V_14/V_13_skew',
            'V_7_min*var440', 'behaday-last_consumer*V_6_last',
            'var6/V_13_last', 'var1*V_6_last', 'ccx_id*behaday-last_consumer',
            'V_3/var13', 'V_3/V_10_skew', 'var17/V_7_skew',
            'consumer_V_7-V_11_median*V_10_std',
            'consumer_V_7-V_11_median/behaday-last_consumer', 'var4/V_14',
            'var1644/V_7_mean', 'var1644/V_10_std', 'var1644*var6',
            'var440/var6', 'ccx_id/basedata_null_rate', 'ccx_id*V_7_skew',
            'ccx_id*var1644', 'V_3*basedata_null_rate', 'V_3*var13',
            'V_3*V_7_std', 'var17/V_10_skew', 'var17*V_13_skew']
temp = raw_train[raw200].join(cv_data1[cv_feat1])
sa30 = ['ccx_id', 'var1*V_7_mean', 'var1/V_7_mean',
        'basedata_null_rate*V_7_mean', 'var1*basedata_null_rate',
        'var1*V_7_min', 'V_13_median/var4', 'var1644/V_7_min',
        'basedata_null_rate*V_7_min', 'V_2*V_13_median', 'var1/V_7_min',
        'var1/var13', 'var17*basedata_null_rate', 'V_2/var1644',
        'var1/var440', 'var753', 'ccx_id/var1272', 'V_7_skew/V_13_skew',
        'var1/ccx_id', 'basedata_null_rate*var13', 'var1644*var440',
        'V_3*var6', 'ccx_id*V_7_mean', 'basedata_null_rate/V_7_min',
        'var1/var1644', 'var437', 'var13/V_13_sum', 'V_13_std/V_13_sum',
        'V_3/var1644', 'ccx_id/V_7_min', 'V_13_median/var1272']
cv_data2 = get_cv_feat(sa30, temp)
cv_feat2 = ['var1/V_7_min/basedata_null_rate*var13',
            'var1/V_7_mean/basedata_null_rate*var13', 'ccx_id/ccx_id/V_7_min',
            'var1644*var440*V_13_std/V_13_sum', 'var1/V_7_mean*var1/V_7_min',
            'ccx_id/var1/ccx_id', 'V_2/var1644/V_13_std/V_13_sum',
            'var1/V_7_mean/var17*basedata_null_rate', 'var1*V_7_min*var437',
            'var1/V_7_min/var17*basedata_null_rate', 'ccx_id/var1/V_7_mean',
            'basedata_null_rate*V_7_mean/basedata_null_rate*V_7_min',
            'var1/ccx_id/var13/V_13_sum', 'V_2/var1644*var1/var1644',
            'var1/V_7_mean*var1*V_7_min', 'var1/V_7_mean*var437',
            'V_13_median/var4*V_3*var6', 'V_2/var1644*var13/V_13_sum',
            'var13/V_13_sum*ccx_id/V_7_min', 'V_2*V_13_median*ccx_id*V_7_mean',
            'V_13_median/var4/V_2*V_13_median', 'var753/V_7_skew/V_13_skew',
            'V_7_skew/V_13_skew*V_13_std/V_13_sum', 'var1/var1644/V_3/var1644',
            'V_13_median/var4/var17*basedata_null_rate',
            'var1/V_7_mean/var1/V_7_min',
            'var17*basedata_null_rate/ccx_id/var1272',
            'V_7_skew/V_13_skew/V_3*var6',
            'V_13_median/var4*V_13_std/V_13_sum',
            'basedata_null_rate*V_7_min/V_3*var6',
            'var17*basedata_null_rate/V_3*var6', 'ccx_id/var1/V_7_min',
            'V_2*V_13_median*V_13_std/V_13_sum',
            'var1/V_7_min*basedata_null_rate/V_7_min', 'V_2/var1644/var753',
            'V_3*var6/var13/V_13_sum', 'var1644/V_7_min*var1/var1644',
            'V_2*V_13_median*V_7_skew/V_13_skew',
            'var17*basedata_null_rate/basedata_null_rate/V_7_min',
            'var1644*var440/V_3*var6', 'var437/ccx_id/V_7_min',
            'ccx_id*var13/V_13_sum',
            'var1*basedata_null_rate*V_13_median/var4',
            'V_13_median/var4*basedata_null_rate/V_7_min',
            'var1644/V_7_min*V_3/var1644', 'V_2*V_13_median/V_3*var6',
            'V_13_std/V_13_sum*V_13_median/var1272',
            'ccx_id/basedata_null_rate*var13',
            'basedata_null_rate*V_7_mean*var1644/V_7_min',
            'V_7_skew/V_13_skew*basedata_null_rate*var13',
            'var1/ccx_id/V_3*var6', 'basedata_null_rate*var13*var13/V_13_sum',
            'V_3*var6*var437', 'basedata_null_rate*V_7_mean/V_13_std/V_13_sum',
            'V_13_median/var4/V_7_skew/V_13_skew',
            'V_2*V_13_median*ccx_id/var1272', 'var1/V_7_min/var1/var13',
            'ccx_id/V_7_skew/V_13_skew', 'V_13_median/var4*V_7_skew/V_13_skew',
            'var1644/V_7_min*var1/V_7_min',
            'var1644/V_7_min/V_13_std/V_13_sum',
            'V_2*V_13_median*V_13_median/var1272', 'var753/V_3*var6',
            'var1/ccx_id/ccx_id*V_7_mean', 'var1/ccx_id*V_13_std/V_13_sum',
            'basedata_null_rate*var13/V_3*var6', 'var1*V_7_mean/var1*V_7_min',
            'basedata_null_rate*V_7_mean*basedata_null_rate*V_7_min',
            'var1*basedata_null_rate/var1/ccx_id',
            'var1*basedata_null_rate*basedata_null_rate*var13',
            'V_13_median/var4*var13/V_13_sum',
            'V_2*V_13_median/V_7_skew/V_13_skew',
            'var17*basedata_null_rate/V_13_std/V_13_sum',
            'var1/ccx_id*V_13_median/var1272',
            'var13/V_13_sum/V_13_std/V_13_sum',
            'ccx_id/basedata_null_rate/V_7_min', 'ccx_id/var13/V_13_sum',
            'var1*V_7_mean/var1/V_7_mean',
            'basedata_null_rate*V_7_mean/var1*basedata_null_rate',
            'var1*V_7_min/V_2*V_13_median', 'V_13_median/var4/var13/V_13_sum',
            'var1/V_7_min*var17*basedata_null_rate',
            'V_2/var1644/ccx_id*V_7_mean', 'V_7_skew/V_13_skew/var13/V_13_sum',
            'V_3*var6*V_13_std/V_13_sum', 'V_3/var1644/V_13_median/var1272',
            'ccx_id/var1*V_7_mean', 'ccx_id/ccx_id*V_7_mean',
            'var1*V_7_mean/V_2*V_13_median', 'var1*V_7_mean*var437',
            'var1/V_7_mean/var1/var13', 'var1*V_7_min/ccx_id/V_7_min',
            'V_13_median/var4*var1644*var440', 'var1/var13*var437',
            'var17*basedata_null_rate*basedata_null_rate/V_7_min',
            'var753*var1/var1644', 'V_7_skew/V_13_skew*var13/V_13_sum',
            'V_3*var6*var1/var1644', 'var1/var1644/V_13_std/V_13_sum',
            'var13/V_13_sum*V_13_std/V_13_sum', 'var1/V_7_mean/V_3*var6']


train = raw_train[raw200].join(cv_data1[cv_feat1])
train = train.join(cv_data2[cv_feat2])
del cv_data1, cv_data2
train['target'] = raw_train['target']
label = 'target'
features = ['var1/V_7_mean/basedata_null_rate*var13',
            'var1/V_7_min/basedata_null_rate*var13',
            'V_2/var1644*var1/var1644', 'V_2/var1644/V_13_std/V_13_sum',
            'var1644*var440*V_13_std/V_13_sum', 'ccx_id/var1/ccx_id',
            'ccx_id/V_7_min', 'ccx_id/ccx_id/V_7_min',
            'var1/V_7_mean*var1/V_7_min',
            'V_13_median/var4*basedata_null_rate/V_7_min', 'V_7_std*V_6_last',
            'var1/V_7_mean*var437', 'var1/V_7_min/var17*basedata_null_rate',
            'var1/V_7_min/var1/var13', 'var1*V_7_min/ccx_id/V_7_min',
            'V_2/var1644/var753', 'var1299', 'basedata_null_rate/var1310',
            'V_2/var1644*var13/V_13_sum', 'V_2*V_13_median*ccx_id/var1272',
            'var11', 'V_2*V_13_median', 'var17*basedata_null_rate/V_3*var6',
            'var1*V_7_mean', 'var1644/V_7_min', 'var1/var440',
            'V_13_std*V_13_last', 'V_14/V_7_min',
            'consumer_V_7-V_11_median/behaday-last_consumer', 'V_3*V_13_sum',
            'V_3/V_13_last', 'consumer_V_7-V_11_median/V_14',
            'var1*V_7_min*var437', 'var1/V_7_min*basedata_null_rate/V_7_min',
            'var1644/V_6_last', 'V_13_median/var4*V_3*var6',
            'basedata_null_rate*V_7_mean*var1644/V_7_min', 'var1/V_13_skew',
            'var1*V_13_skew', 'V_10_std/V_13_sum',
            'var1/ccx_id/var13/V_13_sum', 'V_7_skew/V_13_skew/V_3*var6',
            'V_7_skew/V_13_skew', 'var1/V_7_mean/var17*basedata_null_rate',
            'V_13_median/var4/var17*basedata_null_rate', 'V_6_median',
            'V_7_mean/V_14', 'V_2*V_13_median*ccx_id*V_7_mean',
            'var1*V_7_min/V_2*V_13_median', 'var1644*var440',
            'V_13_median*V_13_sum', 'V_13_median/V_7_std',
            'var17*basedata_null_rate/V_13_std/V_13_sum',
            'V_2/var1644/ccx_id*V_7_mean', 'V_7_std', 'V_14/V_6_last',
            'V_13_skew*behaday-last_consumer', 'V_2*V_13_last',
            'var1*V_6_last', 'V_7_skew/V_13_skew*V_13_std/V_13_sum',
            'var1/var1644/V_13_std/V_13_sum', 'V_5_last', 'V_13_skew',
            'var4/V_13_max', 'V_3/V_7_std', 'var1/V_7_mean*var1*V_7_min',
            'var17*basedata_null_rate/ccx_id/var1272', 'var1292', 'var1295',
            'morning_order_count', 'ccx_id/var1272', 'V_3/V_13_skew',
            'V_13_median/var4/V_2*V_13_median',
            'V_2*V_13_median*V_7_skew/V_13_skew',
            'basedata_null_rate*V_7_mean/V_13_std/V_13_sum', 'var1166',
            'var1*V_7_min', 'var1/ccx_id', 'var13/V_13_sum*ccx_id/V_7_min',
            'var1644*var440/V_3*var6', 'var1/ccx_id/V_3*var6',
            'basedata_null_rate*V_7_mean/var1*basedata_null_rate',
            'V_13_median/var4/var13/V_13_sum', 'consumer_V_7-V_11_median',
            'behaday-last_consumer', 'var13*V_14',
            'V_14*behaday-last_consumer', 'ccx_id/var1/V_7_min',
            'var1644/V_7_min*V_3/var1644', 'var1644/V_7_min/V_13_std/V_13_sum',
            'var1*V_7_mean/V_2*V_13_median', 'ccx_id', 'consumer_V_7-V_11_std',
            'V_3/V_13_sum', 'V_3*var6', 'var1/V_7_skew', 'V_3*V_13_skew',
            'consumer_V_7-V_11_median*V_10_std', 'ccx_id/var1/V_7_mean',
            'basedata_null_rate*V_7_mean/basedata_null_rate*V_7_min',
            'ccx_id/basedata_null_rate*var13', 'var1/var13*var437',
            'weekday_count', 'var1/V_7_mean', 'V_5_last/var1644',
            'var17*V_13_max']


print("test...")
test_B = get_feat_B(test_behavior_B, test_consumer_B)

###cv_feat test
cv_test1 = get_cv_feat(fa30, test_B)
temp = test_B[raw200].join(cv_test1[cv_feat1])
cv_test2 = get_cv_feat(sa30, temp)
test_B = test_B[raw200].join(cv_test1[cv_feat1])
test_B = test_B.join(cv_test2[cv_feat2])
del cv_test1, cv_test2

print("train-shape:", train.shape)
print("test-shape:",test_B.shape)
# train,test_B=get_behaviorpred(train_behavior_A,target,train,test_B,test_behavior_B,flag="B")



print("train....")

params = {
    'boosting_type': 'gbdt',
    'metric': 'auc',
    #     "task":"train",
    'is_unbalance': 'True',
    'learning_rate': 0.01,
    'verbose': 0,
    'num_leaves': 32,
    #             'max_depth':3,
    #             'max_bin':10,
    #             'lambda_l2': 10,
    'objective': 'binary',
    'feature_fraction': 0.4,
    'bagging_fraction': 0.7,  # 0.9是目前最优的
    #             'bagging_freq':1,  # 3是目前最优的
    #             'min_data': 500,
    'seed': 1024,
    'nthread': 12,
    #             'silent': True,
}
dtrain = lgb.Dataset(train[features], train[label])
model = lgb.train(params, dtrain, num_boost_round=400,
                  verbose_eval=200,
                  )
pred = model.predict(test_B[features])
test_B['prob'] = pred
predict_result_B = test_B[['ccx_id', 'prob']]
predict_result_B.to_csv('./predict_result_B.csv',
                        encoding='utf-8', index=False)
print("use time:", datetime.datetime.now()-start, " s")
