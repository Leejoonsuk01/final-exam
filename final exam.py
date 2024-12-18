import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows의 경우
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# 데이터셋 로드
# 파일 경로를 정확히 지정하세요
file_path = 'C:/Users/ljs01/Documents/GitHub/gimal/data/1.smoke_detection_iot.csv'

try:
    # 데이터셋 로드
    data = pd.read_csv(file_path)
    print("데이터셋이 성공적으로 로드되었습니다!")
    print(data.head())  # 데이터 구조 확인
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {file_path}")
    raise
except Exception as e:
    print(f"데이터셋을 로드하는 동안 오류가 발생했습니다: {e}")
    raise

# 데이터 전처리: 특징(features)과 타겟(target) 분리
features = data.drop(columns=["Unnamed: 0", "UTC", "Fire Alarm"])
target = data["Fire Alarm"]

# 학습-테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# 랜덤 포레스트 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# 결과 데이터프레임 생성
results_df = X_test.copy()
results_df['실제 값'] = y_test.values
results_df['예측 값'] = y_pred

# 실제 값 vs 예측 값 바 플롯 저장
fire_alarm_actual = results_df['실제 값'].value_counts()
fire_alarm_predicted = results_df['예측 값'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(['실제 - 화재 없음', '실제 - 화재'], fire_alarm_actual, alpha=0.7, label='실제 값')
plt.bar(['예측 - 화재 없음', '예측 - 화재'], fire_alarm_predicted, alpha=0.7, label='예측 값')
plt.title('실제 값 vs 예측 값 화재 경보')
plt.ylabel('개수')
plt.legend()
actual_vs_predicted_path = 'actual_vs_predicted_fire_alarms.png'
plt.savefig(actual_vs_predicted_path)
plt.close()

# 특징 중요도 플롯 저장
feature_importances = pd.Series(model.feature_importances_, index=features.columns)
feature_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(12, 8))
feature_importances.plot(kind='bar')
plt.title('화재 감지 모델에서의 특징 중요도')
plt.ylabel('중요도')
feature_importance_path = 'feature_importance_plot.png'
plt.savefig(feature_importance_path)
plt.close()

# 결과 데이터셋 저장
results_file_path = 'smoke_detection_results.csv'
results_df.to_csv(results_file_path, index=False, encoding='utf-8-sig')

# 저장된 파일 경로 출력
print("결과 파일이 저장되었습니다:", results_file_path)
print("실제 값 vs 예측 값 플롯이 저장되었습니다:", actual_vs_predicted_path)
print("특징 중요도 플롯이 저장되었습니다:", feature_importance_path)
