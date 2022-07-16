#!/usr/bin/env python
# coding: utf-8

# In[113]:
import catboost
import xgboost
from sklearn.metrics import brier_score_loss, roc_auc_score, f1_score

import numpy as np
import pandas as pd
import tqdm
import socceraction.atomic.vaep.formula as vaepformula
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab

# third party in J League data


# In[114]:


def create_features(spadl_h5, features_h5, labels_h5, games):
    # Create features from original hdf files
    # input : hdf files spadl, features and labels
    # games 
    # output : X, Y 
    xfns = [fs.actiontype,
           fs.actiontype_onehot,
           #fs.bodypart,
           fs.bodypart_onehot, 
           fs.result,
           fs.result_onehot,
           fs.goalscore,
           fs.startlocation,
           fs.endlocation,
           fs.movement,
           fs.space_delta,
           fs.startpolar,
           fs.endpolar,
           fs.team,
           #fs.time,
           fs.time_delta,
           #fs.actiontype_result_onehot
           ]
    nb_prev_actions = 3
    Xcols = fs.feature_column_names(xfns, nb_prev_actions)
    X = []
    for game_id in tqdm.tqdm(games.game_id, desc="selecting features"):
        Xi = pd.read_hdf(features_h5, f"game_{game_id}")
        X.append(Xi[Xcols])
    X = pd.concat(X).reset_index(drop=True)
    
    Y_cols = ["scores", "concedes"]
    Y = []
    for game_id in tqdm.tqdm(games.game_id, desc="selecting features"):
        Yi = pd.read_hdf(labels_h5, f"game_{game_id}")
        Y.append(Yi[Y_cols])
    Y = pd.concat(Y).reset_index(drop=True)
    
    return X, Y


# In[115]:


def convert_J2spadl(J_eventdata):
    # convert J data to spadl
    # input:play * 173 J feature
    # output:play * 25 spadl feature
    spadl_feature = ["game_id",
                 "period_id", 
                 "time_seconds",
                 "start_frame",
                 "timestamp",
                 "team_id",
                 "player_id",
                 "start_x", 
                 "start_y", 
                 "end_x", 
                 "end_y",
                 "type_id",
                 "result_id",
                 "bodypart_id",
                 "action_id",
                 "type_name",
                 "result_name",
                 "bodypart_name", 
                 "player_name",
                 "player_nickname",
                 "jersey_number",
                 "country_id",
                 "country_name", 
                 "extra", 
                 "team_name",
                 "player"]   
    spadl_df = pd.DataFrame(columns=spadl_feature)
    game_id = J_eventdata["試合ID"]
    game_len = len(game_id)
    # sort time sequence
    J_eventdata = J_eventdata.sort_values('絶対時間秒数').reset_index(drop=True)
    
    secondhalf_index = J_eventdata.reset_index().query('アクション名=="後半開始"').index[0]
    period_id = [1] * (secondhalf_index) + [2] * (game_len - secondhalf_index)
    first_start_frame = J_eventdata.loc[0]['フレーム番号']
    first_end_frame = J_eventdata[J_eventdata['アクション名']=='前半終了'].iloc[0]['フレーム番号']
    second_start_frame = J_eventdata.loc[secondhalf_index]['フレーム番号']
    start_frame = J_eventdata['フレーム番号'] - first_start_frame
    start_frame[secondhalf_index:] = start_frame[secondhalf_index:] - (second_start_frame - first_end_frame)
    min2sec = lambda x: (x % 100 + 60 * (x // 100))
    time_seconds = list(map(min2sec, J_eventdata["ハーフ開始相対時間"]))
    ball_x_change = lambda x: (x + 157.5) / 3
    ball_y_change = lambda x: (x + 102) / 3
    ball_x_changed = pd.Series(map(ball_x_change, J_eventdata["ボールＸ"]))
    ball_y_changed = pd.Series(map(ball_y_change, J_eventdata["ボールＹ"]))
    start_x = ball_x_changed
    start_y = ball_y_changed
    end_x = ball_x_changed
    end_x = end_x.shift(-1)
    end_y = ball_y_changed
    end_y = end_y.shift(-1)
    def type_idJ2spadl(x):
        if x==29 or x==30 or x==36 or x==75: # Pass
            x = 0 
        elif x == 45: # Cross
            x = 1
        elif x == 44: # Throw-in
            x = 2
        elif x == 21: # CK
            x = 5
        elif x == 35: # Take on, J : drrible
            x = 7
        elif x == 27 or x ==38: # Foul
            x = 8
        elif x == 74: # Tackle
            x = 9
        elif x == 41: # Interception
            x = 10
        elif x == 15: # Shot
            x = 11
        elif x == 42: # Clearance
            x = 18
        elif x == 50 or x == 73: # Drrible
            x = 21
        elif x == 16: # GK
            x = 22
        else:
            x = -1
    
        return x
    type_id = list(map(type_idJ2spadl, J_eventdata["アクションID"]))
    def type_id2name(x):
        if x==0:
            x = "pass"
        elif x == 1:
            x = "cross"
        elif x == 2:
            x = "throw_in"
        elif x == 5:
            x = "corner_crossed"
        elif x == 7:
            x = "take_on"
        elif x == 8:
            x = "foul"
        elif x == 9:
            x = "tackle"
        elif x == 10:
            x = "interception"
        elif x == 11:
            x = "shot"
        elif x == 18:
            x = "clearance"
        elif x == 21:
            x = "dribble"
        elif x == 22:
            x = "goalkick"
        else:
            x = "other"
        return x
    type_name = list(map(type_id2name, type_id))
    def result_id2name(x):
        if x == 0:
            x = "fail"
        elif x == 1:
            x = "success"
        else:
            x = "other"
        return x
    result_id = J_eventdata["F_成功"]
    result_name = list(map(result_id2name, result_id))
    # set spadl feature
    spadl_df["game_id"]=game_id
    spadl_df["period_id"] = period_id
    spadl_df["time_seconds"] = time_seconds
    spadl_df['start_frame'] = start_frame
    spadl_df["timestamp"] = J_eventdata["ハーフ開始相対時間"]
    spadl_df["team_id"] = J_eventdata["チームID"]
    spadl_df["player_id"] = J_eventdata["選手ID"]
    spadl_df["start_x"] = start_x
    spadl_df["start_y"] = start_y
    spadl_df["end_x"] = end_x
    spadl_df["end_y"] = end_y
    spadl_df["type_id"] = type_id
    spadl_df["result_id"] = J_eventdata["F_成功"]
    spadl_df["action_id"] = range(game_len)
    spadl_df["type_name"] = type_name
    spadl_df["result_name"] = result_name
    spadl_df["player_name"] = J_eventdata["選手名"]
    spadl_df["player_nickname"] = J_eventdata["選手名"]
    spadl_df["jersey_number"] = J_eventdata["選手背番号"]
    spadl_df["team_name"] = J_eventdata["チーム名"]
    spadl_df["player"] = J_eventdata["選手名"]
    
    return spadl_df


# In[116]:


def convert_spadl2train(spadl):
    # convert spadl to train data
    # input:play * 25
    # out:X play * 148 and Y play * 3
    xfns =[fs.actiontype,
      fs.actiontype_onehot,
      #fs.bodypart,
      fs.bodypart_onehot,
      fs.result,
      fs.result_onehot,
      fs.goalscore,
      fs.startlocation,
      fs.endlocation,
      fs.movement,
      fs.space_delta,
      fs.startpolar,
      fs.endpolar,
      fs.team,
      #fs.time,
      fs.time_delta,
      #fs.actiontype_result_onehot    
      ]
    type_id = list(range(23))
    type_name = ["pass",
                "cross",
                "throw_in",
                "freekick_crossed",
                "freekick_shot",
                "corner_crossed",
                "corner_shot",
                "take_on",
                "foul",
                "tackle",
                "interception",
                "shot",
                "shot_penalty",
                "shot_freekick",
                "keeper_save",
                "keeper_claim",
                "keeper_punch",
                "keeper_pick_up",
                "clearance",
                "bad_touch",
                "non_action",
                "dribble",
                "goalkick"]
    actiontypes = pd.DataFrame(columns=["type_id", "type_name"])
    actiontypes["type_id"] = type_id
    actiontypes["type_name"] = type_name
    bodypart_id = list(range(3))
    bodypart_name = ["foot",
                    "head",
                    "other"]
    bodyparts = pd.DataFrame(columns=["bodypart_id", "bodypart_name"])
    bodyparts["bodypart_id"] = bodypart_id
    bodyparts["bodypart_name"] = bodypart_name
    result_id = list(range(6))
    result_name = ["fail",
                  "success",
                  "offside",
                  "owngoal",
                  "yellow_card",
                  "red_card"]
    results = pd.DataFrame(columns=["result_id", "result_name"])
    results["result_id"] = result_id
    results["result_name"] = result_name
    spadl = (spadl.merge(actiontypes, how = "left")
               .merge(results, how = "left")
               .reset_index(drop=True)
               )
    gamestate = fs.gamestates(spadl, 3)
    X = pd.concat([fn(gamestate) for fn in xfns], axis=1)
    nb_prev_actions = 1
    X_cols = fs.feature_column_names(xfns, nb_prev_actions)
    
    yfns = [lab.scores, lab.concedes, lab.goal_from_shot]
    Y = pd.concat([fn(spadl) for fn in yfns], axis=1)
    Y_cols = ["scores", "concedes"]
    
    return X, X_cols, Y, Y_cols


# In[117]:


def train_model(X_train, Y_train):
    # create train model
    # input
    # X_train:play * 148 features
    # Y_train:play * 3 results
    # output
    # models:train model
    
    models_xgb ={}
    for col in Y_train.columns:
        model_xgb = xgboost.XGBClassifier()
        model_xgb.fit(X_train, Y_train[col])
        models_xgb[col] = model_xgb
    
    models_cat = {}
    for col in Y_train.columns:
        model_cat = catboost.CatBoostClassifier(custom_metric="F1")
        model_cat.fit(X_train, Y_train[col])
        models_cat[col] = model_cat
        
    return models_xgb, models_cat


# In[118]:


def estimate_vaep(models, X_test, Y_test, test_spadl):
    # estimate vaep values
    # input 
    # models:train models
    # X_test:play * 148 features
    # Y_test:play * 3 results
    # output 
    # vaep_values:play * 3 values
    Y_col = ["scores", "concedes"]
    Y_hat = pd.DataFrame()
    Y_hat_label = pd.DataFrame()
    print("0.5")
    for col in Y_col:
        Y_hat[col] = [p[1] for p in models[col].predict_proba(X_test)]
        Y_hat_label[col] = np.where(Y_hat[col] > 0.5, 1, 0)
        
        # error handling in case Y_test not in True
        if len(Y_test[Y_test[col]==True])==0:
            continue
        else:
            print("{}".format(col))
            print("ROC AUC:{}".format(roc_auc_score(Y_test[col], Y_hat[col])))
            print("Brier Score:{}".format(brier_score_loss(Y_test[col],Y_hat[col])))
            print("F1 Score:{}".format(f1_score(Y_test[col], Y_hat_label[col])))
    vaep_values = vaepformula.value(test_spadl, Y_hat.scores, Y_hat.concedes)
    
    return vaep_values


# In[119]:


def player_rating(spadl_df, values, player_data):
    # calculate player rating
    # input 
    # spadl_df:play * 25, values:play * 3, player_data:player * 8
    # output
    # player_rating:player * 7
    player_rating = pd.DataFrame(columns=["player_id", 
                                         "team_id", 
                                         "player", 
                                         "vaep_value", 
                                         "count",
                                         "minutes_played",
                                         "vaep_rating"])
    in_player = player_data[player_data.出場==1]
    player_id = in_player["選手ID"]
    team_id = in_player["チームID"]
    player = in_player["選手名"]
    
    player_rating["player_id"] = player_id
    player_rating["team_id"] = team_id
    player_rating["player"] = player
    
    total_data = pd.concat([spadl_df, values], axis=1)
    
    for player in player_rating["player_id"]:
        vaep_sum = sum(total_data[total_data.player_id==player].vaep_value)
        count = len(total_data[total_data.player_id==player].vaep_value)
        player_rating.loc[player_rating.player_id==player, "vaep_value"] = vaep_sum
        player_rating.loc[player_rating.player_id==player, "count"] = count
    
    return player_rating


# In[120]:


def convert_DMatrix(X_train, Y_train):
    # Convert DMatrix for XGBoost
    # input X_train, Y_train:score, concedes
    event_num = len(Y_train["scores"])
    #scores_weight = [len(Y_train[Y_train["scores"]==True]) / event_num , len(Y_train[Y_train["scores"]==False]) / event_num ]
    #concedes_weight = [len(Y_train[Y_train["concedes"]==True]) / event_num , len(Y_train[Y_train["concedes"]==False]) / event_num ]
    
    scores_label = pd.DataFrame()
    scores_label["scores"] = Y_train["scores"]
    #scores_label["no_scores"] = ~Y_train["scores"]
    scores_weight = scores_label * 100 + 1
    
    concedes_label = pd.DataFrame()
    concedes_label["concedes"] = Y_train["concedes"]
    #concedes_label["no_concedes"] = ~Y_train["concedes"]
    concedes_weight = concedes_label * 100 + 1
    
    
    
    dm_train_scores = xgboost.DMatrix(X_train, label=Y_train["scores"], weight = scores_weight)
    dm_train_concedes = xgboost.DMatrix(X_train, label=Y_train["concedes"], weight = concedes_weight)
    
    return dm_train_scores, dm_train_concedes


# In[121]:


def model_train_DMatrix(DMatrix, X_test):
    # input:Dmatrix(weight), X_test(features)
    print("barori")
    params = {
        'objective': 'reg:squarederror','silent':1, 'random_state':1234, 
        # 学習用の指標 (RMSE)
        'eval_metric': 'rmse',
    }
    num_round = 500
    model = xgboost.train(params, DMatrix, num_round)
    dm_test = xgboost.DMatrix(X_test)
    predict = model.predict(dm_test)
    print("predict")
    
    return predict


# In[ ]:




