import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


def negative_sampler(
    df, # implicit feedback(positive)만을 담고 있는 pd.DataFrame
    col_item, # 조합으로 itemID를 정의하는 데 쓰일 칼럼들을 담은 리스트 e.g.) ['product_code','pgm_id'] 
    demo_feats, # user 정보를 담은 칼럼명 리스트 (col_user를 기준으로 merge)
    item_feats, # 상품(case) 정보를 담은 칼럼명 리스트 (col_item으로 생성한 itemID를 기준으로 merge)
    col_user='userID',
    col_user_enroll_time='user_enroll', # 유저의 서비스 가입 시기 || col_item_time과 비교 가능해야 함
    col_item_time='item_time', # item 판매 시기 || col_user_enroll_time과 비교 가능해야 함
    col_label='label',
    ratio_neg_per_user=1,
    use_pandarallel=True,
):

    print('making itemID')
    # item_time 순으로 정렬
    df.sort_values(col_item_time,inplace=True)
    # col_item 정보의 조합으로 새로운 itemID 칼럼 생성
    df.insert(loc=0, column='itemID', value=df.set_index(col_item).index.factorize()[0]+1) 
    
    # 전체 item 개수
    item_n = df['itemID'].nunique()
    
    print('making <user - item_set> df ...')
    itemToList_df = df.groupby(col_user)['itemID'].progress_apply(set).reset_index()

    print('making <user:rated_item_set> dict ...')
    # {userID : 이미 interaction한 item set} dict 생성
    user_rated_dict = pd.Series(itemToList_df['itemID'].values,index=itemToList_df[col_user]).to_dict()
    
    print('making <item:item_time> dict ...')
    # {itemID : item_time} dict 생성
    item_time_dict = pd.Series(df[col_item_time].values,index=df['itemID']).to_dict()
    # 주어진 df에서 가장 작은 item_time 선언
    item_time_min = df.loc[df[col_item_time]>0][col_item_time].min()

    print('making item_time point dict ...')
    itemTimeToList_df = df.groupby(col_item_time)['itemID'].progress_apply(set).reset_index()

    get_min = np.vectorize(lambda x: min(x))
    time_break_dict = pd.Series(get_min(itemTimeToList_df['itemID'].values),
                                index=itemTimeToList_df[col_item_time]
                               ).to_dict()

    print('time_break_dict\n',time_break_dict)
    item_time_zero_to_next_range = range(1,time_break_dict[item_time_min])


    def sample_items(row):
        # Sample negative items for the data frame restricted to a specific user
        
        user_time = row[col_user_enroll_time]

        new_items = []

        if user_time <= item_time_min :
          for _ in range(ratio_neg_per_user):
              t = np.random.randint(1, item_n + 1)
              while t in user_rated_dict[row[col_user]]:
                t = np.random.randint(1, item_n + 1)
              new_items.append((t,item_time_dict[t]))

        else:
          start_time = user_time
          while start_time not in time_break_dict.keys():
            start_time +=1
          itemID_start = time_break_dict[start_time]
          for _ in range(ratio_neg_per_user):
              t = np.random.choice(list(item_time_zero_to_next_range)+ list(range(itemID_start, item_n + 1)))
              while t in user_rated_dict[row[col_user]]:
                t = np.random.choice(list(item_time_zero_to_next_range)+ list(range(itemID_start, item_n + 1)))
              new_items.append((t,item_time_dict[t]))

        return new_items

    res_df = df.copy()

    print('processing negative-sampling ...')
    if use_pandarallel:
      res_df['neg_samples'] = res_df[[col_user,col_user_enroll_time]].parallel_apply(lambda row: sample_items(row),axis=1)
    else:
      res_df['neg_samples'] = res_df[[col_user,col_user_enroll_time]].progress_apply(lambda row: sample_items(row),axis=1)

    print('exploding dataframe ...')
    exploded_df = res_df[[col_user,'neg_samples']].explode('neg_samples')
    if use_pandarallel:
      exploded_df['neg_sample'] = exploded_df['neg_samples'].parallel_apply(lambda x: x[0])
    else:
      exploded_df['neg_sample'] = exploded_df['neg_samples'].progress_apply(lambda x: x[0])
    exploded_df.drop('neg_samples',axis=1,inplace=True)
    exploded_df.reset_index(drop=True,inplace=True)

    print('merging dataframes ...')
    df.drop([col_user_enroll_time, col_item_time],axis=1,inplace=True)

    # merge demographics
    demo_table_for_merge = df[demo_feats].drop_duplicates(col_user)
    neg_df_merge = pd.merge(exploded_df,demo_table_for_merge,on=col_user,how='left')

    # merge case infos
    caseInfo_table_for_merge = df[item_feats+['itemID']].drop_duplicates('itemID')
    neg_df_merge = pd.merge(neg_df_merge,caseInfo_table_for_merge,left_on='neg_sample',right_on='itemID',how='left')

    # drop itemID, neg_sample
    df.drop('itemID',axis=1,inplace=True)
    neg_df_merge.drop(['neg_sample','itemID'],axis=1,inplace=True)

    # assign label
    df[col_label]=1
    neg_df_merge[col_label]=0

    print('making final output ...')
    final_df = pd.concat([df,neg_df_merge]).sort_values(col_user).reset_index(drop=True)

    return final_df