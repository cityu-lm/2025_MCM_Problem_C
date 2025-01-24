import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


medal_counts = pd.read_csv('data/summerOly_medal_counts.csv', encoding='utf-8')
hosts = pd.read_csv('data/summerOly_hosts.csv', encoding='utf-8')
programs = pd.read_csv('data/summerOly_programs.csv', encoding='utf-8')

medal_counts = medal_counts[medal_counts['Year'] <= 2024]
X = medal_counts[['Year', 'Gold', 'Total']]
y_gold = medal_counts['Gold']
y_total = medal_counts['Total']

# 划分数据集和训练集
X_train, X_test, y_gold_train, y_gold_test = train_test_split(X, y_gold, test_size=0.2, random_state=42)
_, _, y_total_train, y_total_test = train_test_split(X, y_total, test_size=0.2, random_state=42)

# 可改进时间序列预测模型
gold_model = LinearRegression()
gold_model.fit(X_train, y_gold_train)

total_model = LinearRegression()
total_model.fit(X_train, y_total_train)

# 评估
gold_predictions = gold_model.predict(X_test)
total_predictions = total_model.predict(X_test)

gold_mse = mean_squared_error(y_gold_test, gold_predictions)
total_mse = mean_squared_error(y_total_test, total_predictions)

print(f'Gold Medal MSE: {gold_mse}')
print(f'Total Medal MSE: {total_mse}')

# 预测2028年的奖牌数：
X_2028 = medal_counts[medal_counts['Year'] == 2024].copy()
X_2028['Year'] = 2028

X_2028['Predicted_Gold_2028'] = gold_model.predict(X_2028[['Year', 'Gold', 'Total']])
X_2028['Predicted_Total_2028'] = total_model.predict(X_2028[['Year', 'Gold', 'Total']])

# 保存预测结果：
X_2028[['NOC', 'Predicted_Gold_2028', 'Predicted_Total_2028']].to_csv('data/summerOly_medal_counts2.csv', mode='a', header=False, index=False)

# 识别可能表现更好或更差的国家：
improvements = X_2028[['NOC', 'Predicted_Gold_2028', 'Predicted_Total_2028']].copy()
improvements = improvements.merge(medal_counts[medal_counts['Year'] == 2024][['NOC', 'Gold', 'Total']], on='NOC')
improvements['Gold_Change'] = improvements['Predicted_Gold_2028'] - improvements['Gold']
improvements['Total_Change'] = improvements['Predicted_Total_2028'] - improvements['Total']

print(improvements.sort_values(by='Gold_Change', ascending=False).head(10))
print(improvements.sort_values(by='Total_Change', ascending=False).head(10))

# 预测将赢得首枚奖牌的国家数量：
first_medal_countries = improvements[(improvements['Gold'] == 0) & (improvements['Predicted_Gold_2028'] > 0)]
print(f'Countries predicted to win their first gold medal: {len(first_medal_countries)}')

# 考虑赛事数量和类型对奖牌数的影响：
event_impact = programs.groupby('Sport').sum().reset_index()
event_impact = event_impact[['Sport', '2024']].rename(columns={'2024': 'Event_Count_2024'})

# Ensure 'NOC' column is present in event_impact before merging

event_impact = event_impact.merge(medal_counts[medal_counts['Year'] == 2024][['NOC', 'Total']], left_on='Sport', right_on='NOC', how='left')

# 打印2024年赛事数量最多的前10个运动项目及其对应的总奖牌数
print(event_impact.sort_values(by='Event_Count_2024', ascending=False).head(10))