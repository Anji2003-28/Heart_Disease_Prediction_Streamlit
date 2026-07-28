import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    df = pd.read_csv('data/heart.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mean_std_values = {'mean': scaler.mean_, 'std': scaler.scale_}
    with open('model/mean_std_values.pkl', 'wb') as f:
        pickle.dump(mean_std_values, f)

    knn = KNeighborsClassifier()
    train_scores = []
    test_scores = []
    for i in range(1, 21):
        knn.set_params(n_neighbors=i)
        knn.fit(X_train, y_train)
        train_scores.append(knn.score(X_train, y_train))
        test_scores.append(knn.score(X_test, y_test))

    best_k = int(np.argmax(test_scores)) + 1
    knn.set_params(n_neighbors=best_k)
    knn.fit(X_train, y_train)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(knn, f)

    print(f'Trained model with k={best_k} and saved model/model.pkl')
