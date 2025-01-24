import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 加载数据
medal_counts = pd.read_csv('data/summerOly_medal_counts.csv', encoding='utf-8')
athletes = pd.read_csv('data/summerOly_athletes.csv', encoding='utf-8')

# 识别著名教练更换国家的实例
notable_coaches = {
    'Lang Ping': {'countries': ['USA', 'CHN'], 'start_years': [2005, 2013], 'end_years': [2008, 2021]},
    'Coach B': {'countries': ['USA'], 'start_years': [2010], 'end_years': [2020]},
    # 根据需要添加更多教练
}

# 分析教练任期前后奖牌数的变化
def analyze_coach_impact(coach_info):
    results = []
    for country, start_year, end_year in zip(coach_info['countries'], coach_info['start_years'], coach_info['end_years']):
        before_tenure = medal_counts[(medal_counts['NOC'] == country) & (medal_counts['Year'] < start_year)]
        during_tenure = medal_counts[(medal_counts['NOC'] == country) & (medal_counts['Year'] >= start_year) & (medal_counts['Year'] <= end_year)]
        after_tenure = medal_counts[(medal_counts['NOC'] == country) & (medal_counts['Year'] > end_year)]
        
        before_medals = before_tenure['Total'].mean() if not before_tenure.empty else 0
        during_medals = during_tenure['Total'].mean() if not during_tenure.empty else 0
        after_medals = after_tenure['Total'].mean() if not after_tenure.empty else 0
        
        results.append((country, before_medals, during_medals, after_medals))
    return results

# 收集建模数据
coach_impact_data = []
for coach, info in notable_coaches.items():
    impacts = analyze_coach_impact(info)
    for country, before, during, after in impacts:
        coach_impact_data.append({
            'coach': coach,
            'country': country,
            'before_medals': before,
            'during_medals': during,
            'after_medals': after
        })

coach_impact_df = pd.DataFrame(coach_impact_data)

# 创建模型以量化优秀教练对奖牌数的影响
X = coach_impact_df[['before_medals']]
y = coach_impact_df['during_medals'] - coach_impact_df['before_medals']

model = LinearRegression()
model.fit(X, y)

# 估算教练效应对总奖牌数的贡献
coach_impact_df['predicted_impact'] = model.predict(coach_impact_df[['before_medals']])
coach_impact_df['actual_impact'] = coach_impact_df['during_medals'] - coach_impact_df['before_medals']

print(coach_impact_df)

# 识别三个国家和运动项目，这些国家和项目中投资优秀教练可能会产生显著影响
countries_sports = [
    {'country': 'USA', 'sport': 'Swimming'},
    {'country': 'CHN', 'sport': 'Gymnastics'},
    {'country': 'GBR', 'sport': 'Athletics'},
    # 根据需要添加更多国家和运动项目
]

# 估算这些国家和运动项目的奖牌数潜在增加量
def estimate_potential_increase(country, sport):
    avg_medals = athletes[(athletes['NOC'] == country) & (athletes['Sport'] == sport)]['Medal'].count() / len(athletes['Year'].unique())
    predicted_increase = model.predict([[avg_medals]])[0]
    return predicted_increase

potential_increases = []
for cs in countries_sports:
    increase = estimate_potential_increase(cs['country'], cs['sport'])
    potential_increases.append({
        'country': cs['country'],
        'sport': cs['sport'],
        'predicted_increase': increase
    })

potential_increases_df = pd.DataFrame(potential_increases)
print(potential_increases_df)
