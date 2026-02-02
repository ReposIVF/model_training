import torch
from torch import nn
from tab_transformer_pytorch import TabTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargando y preparando datos
data_path = 'tmp/motility_morpho_selected_data.csv'
data = pd.read_csv(data_path)

# Codificar la columna objetivo
data['blastocyst_2_encoded'] = (data['blastocyst_2'] == 'Observable ICM and TE').astype(int)

# Seleccionar características continuas
features = ['sta_VSL', 'sta_VCL', 'sta_HMP', 'sta_orientated_angle_mean', 
            'sta_circularity_mean', 'sta_convexity_mean', 'sta_compactness_mean', 
            'sta_minor_axis_radius_mean']

# Dividir los datos en entrenamiento y prueba de manera estratificada
X_train, X_test, y_train, y_test = train_test_split(
    data[features], 
    data['blastocyst_2_encoded'], 
    test_size=0.2, 
    random_state=42#, 
   # stratify=data['blastocyst_2_encoded']
)

# Configuración del modelo TabTransformer
model = TabTransformer(
    categories = tuple(),  # Sin categorías, solo características continuas
    num_continuous = len(features),
    dim = 32,
    dim_out = 1,
    depth = 6,
    heads = 8,
    attn_dropout = 0.1,
    ff_dropout = 0.1,
    mlp_hidden_mults = (4, 2),
    mlp_act = nn.ReLU()
)

# Función para calcular y mostrar métricas
def calculate_metrics(y_true, y_pred):
    y_pred_label = (y_pred > 0.5).float()  # Umbral de 0.5 para convertir a clases binarias
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

# Conversión de los datos a tensor
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

# Conversión de los datos de prueba a tensor
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Configuración del optimizador y la función de pérdida
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Configuración de early stopping basado en accuracy
early_stopping_patience = 20
best_accuracy = 0
epochs_no_improve = 0
best_model = None

# Variable para la frecuencia de evaluación de métricas
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

# Ciclo de entrenamiento con monitoreo de métricas y early stopping
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    
    pred_train = model(torch.empty(X_train_tensor.size(0), 0).to(X_train_tensor.device), X_train_tensor)
    loss = criterion(pred_train, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % eval_every == 0 or epoch == 999:  # Utilizar la variable para evaluar cada 'eval_every' épocas
        model.eval()
        
        # Predicciones en el conjunto de entrenamiento
        with torch.no_grad():
            train_accuracy, train_precision, train_recall, train_f1, train_auc = calculate_metrics(y_train_tensor.cpu(), pred_train.cpu())
        
        # Predicciones en el conjunto de prueba
        with torch.no_grad():
            pred_test = model(torch.empty(X_test_tensor.size(0), 0).to(X_test_tensor.device), X_test_tensor)
            test_accuracy, test_precision, test_recall, test_f1, test_auc = calculate_metrics(y_test_tensor.cpu(), pred_test.cpu())
        
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
    torch.save(best_model, 'best_model.pth')
    print(f'Model saved with best Accuracy: {best_accuracy:.4f}')

# Generar las gráficas de las métricas
plot_metrics(history)