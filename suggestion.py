import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold, 
    GridSearchCV, learning_curve, train_test_split
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    make_scorer
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

from basicConfig import X, y

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
RANDOM_STATE = 42
N_SPLITS = 5

# Verificar se X e y estão definidos
try:
    print(f"Dimensões dos dados:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Distribuição: {np.bincount(y)}")
    print(f"Razão p/n: {X.shape[1]/X.shape[0]:.2f}")
    print(f"Baseline (chutar classe majoritária): {max(np.bincount(y))/len(y):.3f}")
except:
    raise ValueError("X e y não estão definidos. Carregue seus dados primeiro!")

# =============================================================================
# 1. ANÁLISE DO DESEQUILÍBRIO
# =============================================================================
print("\n" + "="*60)
print("ANÁLISE DO DESEQUILÍBRIO DE CLASSES")
print("="*60)
print(f"Classe 0: {np.bincount(y)[0]} amostras ({np.bincount(y)[0]/len(y)*100:.1f}%)")
print(f"Classe 1: {np.bincount(y)[1]} amostras ({np.bincount(y)[1]/len(y)*100:.1f}%)")
print(f"⚠️  Dataset EXTREMAMENTE desbalanceado (ratio 1:{np.bincount(y)[1]/np.bincount(y)[0]:.1f})")
print("Acurácia NÃO é uma métrica adequada! Use ROC-AUC, F1, Precision-Recall")

# =============================================================================
# 2. PIPELINES CORRETOS (COM DATA LEAKAGE RESOLVIDO)
# =============================================================================

def create_pipeline_without_leakage(name, use_smote=True):
    """Cria pipeline que evita data leakage usando imblearn Pipeline"""
    
    if name == 'PCA+RF':
        steps = [
            ('scaler', RobustScaler()),  # RobustScaler é melhor com outliers
            ('pca', PCA(n_components=0.95, random_state=RANDOM_STATE)),
        ]
        if use_smote:
            steps.append(('smote', SMOTE(random_state=RANDOM_STATE)))
        steps.append(('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )))
        
    elif name == 'Select+RF_FIXED':
        steps = [
            ('scaler', RobustScaler()),
            ('feature_selection', SelectKBest(f_classif, k=200)),  # Fixo, sem data leakage
        ]
        if use_smote:
            steps.append(('smote', SMOTE(random_state=RANDOM_STATE)))
        steps.append(('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )))
        
    elif name == 'L1_Logistic':
        steps = [
            ('scaler', RobustScaler()),
            ('feature_selection', SelectFromModel(
                LogisticRegression(
                    penalty='l1',
                    solver='saga',
                    C=0.1,
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    class_weight='balanced'
                ),
                threshold='mean'
            )),
        ]
        if use_smote:
            steps.append(('smote', SMOTE(random_state=RANDOM_STATE)))
        steps.append(('classifier', LogisticRegression(
            penalty='l1',
            solver='saga',
            C=0.1,
            max_iter=5000,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )))
        
    elif name == 'LDA+RF':
        steps = [
            ('scaler', RobustScaler()),
            ('lda', LinearDiscriminantAnalysis(n_components=1)),  # Máximo 1 com 2 classes
        ]
        if use_smote:
            steps.append(('smote', SMOTE(random_state=RANDOM_STATE)))
        steps.append(('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )))
    
    return ImbPipeline(steps) if use_smote else Pipeline(steps)

# Criar pipelines
pipelines = {
    'PCA+RF': create_pipeline_without_leakage('PCA+RF', use_smote=True),
    'Select+RF_FIXED': create_pipeline_without_leakage('Select+RF_FIXED', use_smote=True),
    'L1_Logistic': create_pipeline_without_leakage('L1_Logistic', use_smote=True),
    'LDA+RF': create_pipeline_without_leakage('LDA+RF', use_smote=True),
}

# =============================================================================
# 3. VALIDAÇÃO CRUZADA COM MÉTRICAS APROPRIADAS
# =============================================================================

print("\n" + "="*60)
print("VALIDAÇÃO CRUZADA COM MÉTRICAS PARA DADOS DESBALANCEADOS")
print("="*60)

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Métricas apropriadas para desbalanceamento
scoring_metrics = {
    'ROC_AUC': 'roc_auc',
    'F1': make_scorer(f1_score),
    'Precision': make_scorer(precision_score),
    'Recall': make_scorer(recall_score),
    'Avg_Precision': make_scorer(average_precision_score)
}

results = {}

for name, pipeline in pipelines.items():
    print(f"\n{'='*40}")
    print(f"Avaliando: {name}")
    print(f"{'='*40}")
    
    # Cross-validation com múltiplas métricas
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring_metrics,
        n_jobs=-1,
        return_train_score=True
    )
    
    results[name] = {}
    
    for metric in scoring_metrics.keys():
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        results[name][metric] = {
            'test_mean': test_scores.mean(),
            'test_std': test_scores.std(),
            'train_mean': train_scores.mean(),
            'gap': train_scores.mean() - test_scores.mean()
        }
        
        print(f"  {metric:20s}: Test={test_scores.mean():.3f} (+/- {test_scores.std()*2:.3f}) "
              f"| Train={train_scores.mean():.3f} | Gap={train_scores.mean()-test_scores.mean():.3f}")
        
        if train_scores.mean() - test_scores.mean() > 0.1:
            print(f"    ⚠️  Gap alto - possível overfitting")

# =============================================================================
# 4. NESTED CROSS-VALIDATION
# =============================================================================

print("\n" + "="*60)
print("NESTED CROSS-VALIDATION (Validação Rigorosa)")
print("="*60)

inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Grid para otimização
param_grid_rf = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [5, 10, 15],
    'classifier__min_samples_split': [5, 10],
}

for name in ['PCA+RF', 'L1_Logistic', 'LDA+RF']:
    if name in pipelines:
        print(f"\nNested CV para: {name}")
        
        grid = GridSearchCV(
            pipelines[name],
            param_grid_rf if 'RF' in name else {'classifier__C': [0.01, 0.1, 1.0]},
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        nested_scores = cross_validate(
            grid,
            X, y,
            cv=outer_cv,
            scoring=scoring_metrics,
            n_jobs=-1,
            return_train_score=True
        )
        
        for metric in ['test_ROC_AUC', 'test_F1']:
            scores = nested_scores[metric]
            print(f"  {metric.replace('test_', '')}: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# =============================================================================
# 5. ANÁLISE DE OVERFITTING COM CURVAS DE APRENDIZADO
# =============================================================================

print("\nGerando curvas de aprendizado...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, (name, pipeline) in enumerate(pipelines.items()):
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X, y,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='roc_auc'
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    
    axes[idx].fill_between(train_sizes, train_mean - train_std,
                           train_mean + train_std, alpha=0.1, color='blue')
    axes[idx].fill_between(train_sizes, test_mean - test_std,
                           test_mean + test_std, alpha=0.1, color='orange')
    axes[idx].plot(train_sizes, train_mean, 'o-', color='blue', 
                   label='Treino', linewidth=2)
    axes[idx].plot(train_sizes, test_mean, 'o-', color='orange', 
                   label='Validação', linewidth=2)
    
    axes[idx].set_xlabel('Tamanho do Treino')
    axes[idx].set_ylabel('ROC-AUC')
    axes[idx].set_title(f'{name}')
    axes[idx].legend(loc='lower right')
    axes[idx].grid(True, alpha=0.3)
    
    # Calcular gap
    gap = train_mean[-1] - test_mean[-1]
    axes[idx].text(0.5, 0.05, f'Gap: {gap:.3f}', 
                   transform=axes[idx].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Curvas de Aprendizado (ROC-AUC)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# =============================================================================
# 6. VALIDAÇÃO FINAL COM SPLIT TREINO/TESTE
# =============================================================================

print("\n" + "="*60)
print("VALIDAÇÃO FINAL - TREINO/TESTE SPLIT")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Encontrar melhor modelo (maior ROC-AUC médio)
best_model_name = max(results, key=lambda x: results[x]['ROC_AUC']['test_mean'])
print(f"Melhor modelo: {best_model_name}")

# Treinar e avaliar
best_pipeline = pipelines[best_model_name]
best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)
y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

print(f"\nMétricas no conjunto de teste:")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"Avg Precision: {average_precision_score(y_test, y_pred_proba):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Classe 0', 'Classe 1']))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Classe 0', 'Classe 1'],
            yticklabels=['Classe 0', 'Classe 1'])
plt.title(f'Matriz de Confusão - {best_model_name}')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()
plt.show()

# =============================================================================
# 7. ANÁLISE DE FEATURES IMPORTANTES
# =============================================================================

print("\n" + "="*60)
print("ANÁLISE DE FEATURES")
print("="*60)

if best_model_name == 'PCA+RF':
    n_components = best_pipeline.named_steps['pca'].n_components_
    print(f"PCA reduziu de {X.shape[1]} para {n_components} componentes")
    
elif best_model_name == 'Select+RF_FIXED':
    n_selected = best_pipeline.named_steps['feature_selection'].get_support().sum()
    print(f"SelectKBest selecionou {n_selected} de {X.shape[1]} features")
    
elif best_model_name == 'L1_Logistic':
    n_selected = best_pipeline.named_steps['feature_selection'].get_support().sum()
    print(f"L1 regularização selecionou {n_selected} de {X.shape[1]} features")

# Feature importance se disponível
if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'):
    importances = best_pipeline.named_steps['classifier'].feature_importances_
    top_indices = np.argsort(importances)[-10:]
    print("\nTop 10 features mais importantes:")
    for i in reversed(top_indices):
        print(f"  Feature {i}: {importances[i]:.4f}")

# =============================================================================
# 8. RESUMO E RECOMENDAÇÕES
# =============================================================================

print("\n" + "="*60)
print("RESUMO E RECOMENDAÇÕES")
print("="*60)

print(f"\nDataset: {X.shape[0]} amostras, {X.shape[1]} features")
print(f"Classes: Desbalanceadas ({np.bincount(y)[0]}:{np.bincount(y)[1]})")

print("\nResultados Comparativos (ROC-AUC):")
for name in results:
    roc_auc = results[name]['ROC_AUC']
    print(f"  {name:20s}: {roc_auc['test_mean']:.3f} (+/- {roc_auc['test_std']*2:.3f}) "
          f"[gap: {roc_auc['gap']:.3f}]")

print("\n⚠️  Recomendações para seu caso específico:")
print("1. Use SEMPRE ROC-AUC ou F1, NUNCA acurácia com dados desbalanceados")
print("2. O modelo L1_Logistic é mais interpretável e rápido")
print("3. Considere coletar mais dados da classe minoritária (Classe 0)")
print("4. Experimente técnicas avançadas:")
print("   - SMOTE-ENN (combina oversampling + undersampling)")
print("   - Ensemble com bagging focado na classe minoritária")
print("   - Cost-sensitive learning (ajustar pesos das classes)")
print("5. Valide sempre com nested cross-validation")
print("6. Um ROC-AUC > 0.95 já é excelente para aplicações reais")