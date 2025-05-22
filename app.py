import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка и кэширование данных
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_name'] = df['target'].apply(lambda x: iris.target_names[x])
    return df, iris

# Инициализация и кэширование модели
@st.cache_resource
def train_model(X, y, n_estimators, max_depth, criterion):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=criterion,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test

def main():
    st.title('Интерактивная демонстрация модели Random Forest')
    st.write("""
    ## Классификация цветов ириса
    Изменяйте параметры модели и наблюдайте за изменением точности предсказаний
    """)

    # Загрузка данных
    df, iris = load_data()
    X = df[iris.feature_names]
    y = df['target']

    # Сайдбар с настройками модели
    with st.sidebar:
        st.header("Настройки модели")
        n_estimators = st.slider(
            "Количество деревьев", 
            min_value=1, 
            max_value=200, 
            value=100,
            help="Выберите количество деревьев в ансамбле"
        )
        max_depth = st.slider(
            "Максимальная глубина", 
            min_value=1, 
            max_value=30, 
            value=5,
            help="Ограничение глубины каждого дерева"
        )
        criterion = st.selectbox(
            "Критерий разделения",
            ['gini', 'entropy'],
            index=0,
            help="Выберите критерий для разделения узлов"
        )

    # Обучение модели
    model, X_test, y_test = train_model(X, y, n_estimators, max_depth, criterion)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Визуализация результатов
    st.subheader("Оценка производительности модели")
    st.metric("Точность предсказаний", f"{accuracy:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Матрица ошибок")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            ax=ax
        )
        ax.set_xlabel('Предсказанные')
        ax.set_ylabel('Истинные')
        st.pyplot(fig)

    with col2:
        st.write("#### Важность признаков")
        importances = pd.Series(model.feature_importances_, index=iris.feature_names)
        fig, ax = plt.subplots()
        importances.sort_values().plot.barh(ax=ax)
        ax.set_xlabel('Важность')
        st.pyplot(fig)

    # Отчет классификации
    st.write("#### Детальный отчет")
    report = classification_report(y_test, y_pred, target_names=iris.target_names)
    st.text(report)

    # Визуализация исходных данных
    st.subheader("Исследование исходных данных")
    selected_feature = st.selectbox(
        "Выберите признак для визуализации",
        iris.feature_names
    )
    fig, ax = plt.subplots()
    sns.boxplot(x='target_name', y=selected_feature, data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
