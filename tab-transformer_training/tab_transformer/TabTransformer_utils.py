from typing import Tuple
import pandas as pd
import numpy as np
from tab_transformer_pytorch import TabTransformer
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tab_transformer_pytorch import TabTransformer
import torch
from torch import nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def calculate_metrics(y_true, y_pred):
    y_pred_label = (y_pred > 0.5).astype(int)  # Umbral de 0.5 para convertir a clases binarias
    accuracy = accuracy_score(y_true, y_pred_label)
    precision = precision_score(y_true, y_pred_label)
    recall = recall_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label)
    auc = roc_auc_score(y_true, y_pred)
    
    return accuracy, precision, recall, f1, auc

# Función para graficar las métricas
def plot_metrics(history):
    epochs = range(len(history['train_accuracy']))
    
    # Plot accuracy
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['test_accuracy'], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs')

    # Plot precision
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_precision'], label='Train Precision')
    plt.plot(epochs, history['test_precision'], label='Test Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Precision Over Epochs')

    # Plot recall
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_recall'], label='Train Recall')
    plt.plot(epochs, history['test_recall'], label='Test Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.title('Recall Over Epochs')

    # Plot F1-score
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['train_f1'], label='Train F1-Score')
    plt.plot(epochs, history['test_f1'], label='Test F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.title('F1-Score Over Epochs')
    
    plt.tight_layout()
    plt.show()

def feature_dt_prepreprocessing(segmentation_results: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing of dataframe to get dataset and X Y labels
    """
    segmentation_results["label"] = data["pgt"]
    segmentation_results['label'] = segmentation_results['label'].map({'mosaic': 1, 'euploid': 1, 'aneuploid': 0})
    numeric_columns = segmentation_results.select_dtypes(include=["number"]).columns
    segmentation_results[numeric_columns] = segmentation_results[numeric_columns].fillna(0)
    X = segmentation_results.drop(columns=['image', 'label'])
    y = segmentation_results['label']
    return X, y

def get_top_features(X, y, segmentation_results):
    euploid_data = X[y == 1]
    aneuploid_data = X[y == 0]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_scaled, y)
    rf_importances = pd.Series(rf_model.feature_importances_, index=X_scaled.columns).sort_values(ascending=False)
    numerical_data = segmentation_results.select_dtypes(include=[np.number])
    correlation_with_target = numerical_data.corr()['label']
    rfe_model = LogisticRegression(max_iter=1000, random_state=42)
    rfe_selector = RFE(estimator=rfe_model, n_features_to_select=15, step=1)
    rfe_selector.fit(X_scaled, y)
    rfe_ranking = pd.Series(rfe_selector.ranking_, index=X_scaled.columns).sort_values()
    feature_evaluation = pd.DataFrame({
        'Random_Forest_Importance': rf_importances,
        'Correlation_With_Target': correlation_with_target[X_scaled.columns],
        'RFE_Ranking': rfe_ranking
    })
    feature_evaluation['RFE_Score'] = 1 - (feature_evaluation['RFE_Ranking'] - 1) / (feature_evaluation['RFE_Ranking'].max() - 1)
    feature_evaluation['Aggregate_Score'] = (
        feature_evaluation['Random_Forest_Importance'] +
        abs(feature_evaluation['Correlation_With_Target']) +
        feature_evaluation['RFE_Score']
    )
    feature_evaluation = feature_evaluation.sort_values(by='Aggregate_Score', ascending=False)
    top_features = feature_evaluation.head(15).index
    X_selected = X_scaled[top_features]
    return X_selected

def get_transformer_model():
    model = TabTransformer(
        categories=tuple(),  # Sin categorías, solo características continuas
        num_continuous=15,  # 15 características continuas
        dim=256,  # Incrementar dimensión de embedding para capturar más información
        dim_out=1,  # Salida continua
        depth=4,  # Reducir profundidad a 4 para evitar sobreajuste
        heads=2,  # Reducir cabezas a 2 por tamaño moderado de datos
        attn_dropout=0.2,  # Menor dropout para permitir más aprendizaje
        ff_dropout=0.2,  # Reducir dropout para conexiones feed-forward
        mlp_hidden_mults=(4, 2),  # Ajustar dimensiones de la capa MLP
        mlp_act=nn.ReLU()   # Activación no lineal estándar
    )
    return model

def train_model(X_selected, X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=90, stratify=y
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=90, stratify=y
    )
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10) 
    criterion = nn.SmoothL1Loss(beta=1.0)
    early_stopping_patience = 1000
    best_accuracy = 0
    epochs_no_improve = 0
    best_model = None
    eval_every = 10  # Evalúa y muestra métricas cada 10 épocas

    # Historial de métricas
    history = {
        'train_accuracy': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'test_accuracy': [],
        'test_precision': [],
        'test_recall': [],
        'test_f1': []
    }

    for epoch in range(1500):
        model.train()
        optimizer.zero_grad()
        
        pred_train = model(torch.empty(X_train_tensor.size(0), 0).to("cuda"), X_train_tensor)
        loss = criterion(pred_train, y_train_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % eval_every == 0 or epoch == 1499:  # Utilizar la variable para evaluar cada 'eval_every' épocas
            model.eval()
            
            # Predicciones en el conjunto de entrenamiento
            with torch.no_grad():
                train_accuracy, train_precision, train_recall, train_f1, train_auc = calculate_metrics(y_train_tensor.cpu().numpy(), pred_train.cpu().numpy())
            
            # Predicciones en el conjunto de prueba
            with torch.no_grad():
                pred_test = model(torch.empty(X_test_tensor.size(0), 0).to("cuda"), X_test_tensor)
                test_accuracy, test_precision, test_recall, test_f1, test_auc = calculate_metrics(y_test_tensor.cpu().numpy(), pred_test.cpu().numpy())
            
            # Guardar las métricas en el historial
            history['train_accuracy'].append(train_accuracy)
            history['train_precision'].append(train_precision)
            history['train_recall'].append(train_recall)
            history['train_f1'].append(train_f1)
            history['test_accuracy'].append(test_accuracy)
            history['test_precision'].append(test_precision)
            history['test_recall'].append(test_recall)
            history['test_f1'].append(test_f1)
            
            # Mostrar las métricas
            print(f'Epoch {epoch}:')
            print(f'  Training Loss: {loss.item():.4f}')
            print(f'  Training Metrics: Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}, AUC: {train_auc:.4f}')
            print(f'  Test Metrics: Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-Score: {test_f1:.4f}, AUC: {test_auc:.4f}')
            
            # Verificar si se encontró un nuevo mejor modelo basado en accuracy
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                epochs_no_improve = 0
                best_model = model.state_dict()  # Guardar el estado del mejor modelo
                print(f'  New best model found! Accuracy: {best_accuracy:.4f}')
            else:
                epochs_no_improve += 1
                
            # Early stopping
            if epochs_no_improve == early_stopping_patience:
                print(f'Early stopping triggered after {epoch} epochs.')
                break

    # Guardar el mejor modelo
    if best_model is not None:
        torch.save(best_model, 'Care_fertility_finetuned_best_model.pth')
        print(f'Model saved with best Accuracy: {best_accuracy:.4f}')

    # Generar las gráficas de las métricas
    plot_metrics(history)