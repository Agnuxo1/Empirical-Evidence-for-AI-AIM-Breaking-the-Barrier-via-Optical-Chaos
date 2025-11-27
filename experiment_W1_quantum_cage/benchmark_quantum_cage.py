"""
Benchmark para el Experimento W1: La Jaula Cuántica

Este script ejecuta el experimento completo para evaluar si los modelos de IA pueden
desarrollar representaciones que trasciendan las variables humanas tradicionales
en un sistema cuántico simple.

CREDITS AND REFERENCES:
-----------------------
Darwin's Cage Theory:
- Theory Creator: Gideon Samid
- Reference: Samid, G. (2025). Negotiating Darwin's Barrier: Evolution Limits Our View of Reality, AI Breaks Through. Applied Physics Research, 17(2), 102. https://doi.org/10.5539/apr.v17n2p102
- Publication: Applied Physics Research; Vol. 17, No. 2; 2025. ISSN 1916-9639 E-ISSN 1916-9647. Published by Canadian Center of Science and Education
- Available at: https://www.researchgate.net/publication/396377476_Negotiating_Darwin's_Barrier_Evolution_Limits_Our_View_of_Reality_AI_Breaks_Through

Experiments, AI Models, Architectures, and Reports:
- Author: Francisco Angulo de Lafuente
- Responsibilities: Experimental design, AI model creation, architecture development, results analysis, and report writing
"""
import os
import numpy as np
import torch
import torch.nn as nn
from quantum_cage import (
    QuantumDoubleWell, 
    QuantumCageModel, 
    train_model, 
    analyze_representations,
    plot_results,
    plot_potential,
    plot_observables,
    plot_losses
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuración de estilo
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Crear directorio para resultados
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

def generate_training_data(n_samples=1000, n_points=128, t_max=5.0, n_steps=100):
    """Genera datos de entrenamiento para el modelo."""
    print("Generando datos de entrenamiento...")
    
    # Crear el simulador cuántico
    qm = QuantumDoubleWell(n_points=n_points)
    
    X = []
    y = []
    observables = []
    
    for _ in tqdm(range(n_samples), desc="Muestras"):
        # Estado inicial aleatorio (superposición de estados gaussianos)
        n_peaks = np.random.randint(1, 4)
        psi0 = np.zeros(n_points, dtype=complex)
        
        for _ in range(n_peaks):
            x0 = np.random.uniform(-qm.width/2, qm.width/2)
            sigma = np.random.uniform(0.2, 1.0)
            phase = np.random.uniform(0, 2*np.pi)
            
            gaussian = np.exp(-(qm.x - x0)**2 / (4 * sigma**2) + 1j * phase)
            psi0 += gaussian
        
        # Normalizar el estado
        psi0 = psi0 / np.sqrt(np.trapz(np.abs(psi0)**2, dx=qm.dx))
        
        # Generar trayectoria temporal
        trajectory, t_points = qm.generate_trajectory(psi0, t_max, n_steps)
        
        # Calcular observables
        obs = [qm.measure_observables(psi) for psi in trajectory]
        
        # Agregar a los conjuntos de datos
        for i in range(n_steps - 1):
            X.append(trajectory[i])
            y.append(trajectory[i+1])
            observables.append(obs[i])
    
    return np.array(X), np.array(y), observables, qm


def train_quantum_model(X, y, input_dim, epochs=1000, lr=1e-3):
    """Entrena el modelo cuántico."""
    print("\nEntrenando el modelo cuántico...")
    
    # Dividir en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convertir a tensores de PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.complex64)
    y_train_tensor = torch.tensor(y_train, dtype=torch.complex64)
    X_val_tensor = torch.tensor(X_val, dtype=torch.complex64)
    y_val_tensor = torch.tensor(y_val, dtype=torch.complex64)
    
    # Crear el modelo
    model = QuantumCageModel(input_dim)
    
    # Entrenar el modelo
    trained_model, train_losses, val_losses = train_model(
        model, 
        (X_train_tensor, y_train_tensor),
        (X_val_tensor, y_val_tensor),
        epochs=epochs,
        lr=lr
    )
    
    return trained_model, train_losses, val_losses


def train_classical_model(X, y):
    """Entrena un modelo clásico de referencia."""
    print("\nEntrenando modelo clásico de referencia...")
    
    # Convertir a representación de amplitud y fase
    X_amp = np.abs(X)
    X_phase = np.angle(X)
    X_classical = np.column_stack([X_amp, X_phase])
    
    # Mismo procedimiento para las etiquetas
    y_amp = np.abs(y)
    y_phase = np.angle(y)
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_amp_train, y_amp_test, y_phase_train, y_phase_test = train_test_split(
        X_classical, y_amp, y_phase, test_size=0.2, random_state=42
    )
    
    # Modelo para la amplitud
    amp_model = Ridge(alpha=1.0)
    amp_model.fit(X_train, y_amp_train)
    amp_score = amp_model.score(X_test, y_amp_test)
    
    # Modelo para la fase (usamos una red neuronal para capturar no linealidades)
    phase_model = nn.Sequential(
        nn.Linear(X_train.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, y_phase.shape[1])
    )
    
    # Convertir a tensores
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_phase_train_tensor = torch.tensor(y_phase_train, dtype=torch.float32)
    
    # Entrenar el modelo de fase
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(phase_model.parameters(), lr=1e-3)
    
    for epoch in tqdm(range(100), desc="Entrenando modelo de fase"):
        optimizer.zero_grad()
        outputs = phase_model(X_train_tensor)
        loss = criterion(outputs, y_phase_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Evaluar el modelo de fase
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_phase_pred = phase_model(X_test_tensor).numpy()
    
    phase_score = r2_score(y_phase_test, y_phase_pred)
    
    print(f"R² Score - Amplitud: {amp_score:.4f}, Fase: {phase_score:.4f}")
    
    return amp_model, phase_model, amp_score, phase_score


def analyze_cage_breaking(model, X, qm):
    """Analiza si el modelo ha roto la 'jaula' de variables humanas."""
    print("\nAnalizando la 'jaula' de variables humanas...")
    
    # Extraer representaciones internas
    pca_result, pca = analyze_representations(model, X)
    
    # Calcular variables clásicas para comparación
    positions = []
    momenta = []
    for psi in X:
        obs = qm.measure_observables(psi)
        positions.append(obs['position'])
        momenta.append(obs['momentum'])
    
    positions = np.array(positions)
    momenta = np.array(momenta)
    
    # Visualizar las representaciones
    plt.figure(figsize=(12, 10))
    
    # Gráfico de las dos primeras componentes principales
    plt.subplot(2, 2, 1)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=positions, cmap='viridis', alpha=0.6)
    plt.colorbar(label='Posición Esperada')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('PCA de las Representaciones Internas')
    
    # Varianza explicada
    plt.subplot(2, 2, 2)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.title('Varianza Explicada por las Componentes Principales')
    plt.grid(True)
    
    # Correlación con posición y momento
    plt.subplot(2, 2, 3)
    plt.scatter(positions, pca_result[:, 0], alpha=0.6)
    plt.xlabel('Posición Esperada')
    plt.ylabel('Componente Principal 1')
    plt.title('Correlación con la Posición')
    
    plt.subplot(2, 2, 4)
    plt.scatter(momenta, pca_result[:, 1], alpha=0.6, color='orange')
    plt.xlabel('Momento Esperado')
    plt.ylabel('Componente Principal 2')
    plt.title('Correlación con el Momento')
    
    plt.tight_layout()
    plt.savefig('figures/cage_analysis.png')
    plt.close()
    
    # Calcular métricas cuantitativas
    from scipy.stats import pearsonr
    
    # Correlación entre componentes principales y variables clásicas
    corr_pos_pc1 = pearsonr(positions, pca_result[:, 0])[0]
    corr_mom_pc2 = pearsonr(momenta, pca_result[:, 1])[0]
    
    print(f"Correlación PC1 - Posición: {corr_pos_pc1:.4f}")
    print(f"Correlación PC2 - Momento: {corr_mom_pc2:.4f}")
    
    # Evaluar si las representaciones están correlacionadas con variables clásicas
    if abs(corr_pos_pc1) > 0.7 or abs(corr_mom_pc2) > 0.7:
        print("\n🔍 El modelo parece estar reconstruyendo variables clásicas (jaula NO rota).")
        cage_status = "🔒 LOCKED"
    else:
        print("\n✨ El modelo ha desarrollado representaciones que trascienden las variables clásicas (jaula ROTA).")
        cage_status = "🔓 BROKEN"
    
    return {
        'cage_status': cage_status,
        'corr_position': corr_pos_pc1,
        'corr_momentum': corr_mom_pc2,
        'explained_variance': pca.explained_variance_ratio_
    }


def run_experiment():
    """Ejecuta el experimento completo."""
    # Generar datos (reducimos el número de muestras y puntos para hacerlo más rápido)
    print("\n🔧 Configuración del experimento:")
    print("- Muestras: 100 (antes 500)")
    print("- Puntos por muestra: 64 (antes 128)")
    print("- Épocas: 50 (antes 500)")
    print("\n🔄 Generando datos de entrenamiento...")
    
    X, y, observables, qm = generate_training_data(n_samples=100, n_points=64)
    
    # Entrenar modelos
    print("\n🚀 Entrenando modelo cuántico...")
    quantum_model, train_losses, val_losses = train_quantum_model(
        X, y, input_dim=X.shape[1], epochs=50, lr=1e-3
    )
    
    # Entrenar modelo clásico de referencia
    amp_model, phase_model, amp_score, phase_score = train_classical_model(X, y)
    
    # Analizar si el modelo ha roto la 'jaula'
    analysis_results = analyze_cage_breaking(quantum_model, X, qm)
    
    # Visualizar resultados
    plot_losses(train_losses, val_losses, "Pérdidas del Modelo Cuántico")
    
    # Mostrar resumen de resultados
    print("\n" + "="*50)
    print("RESUMEN DEL EXPERIMENTO")
    print("="*50)
    print(f"Estado de la 'Jaula': {analysis_results['cage_status']}")
    print(f"\nMétricas de Calidad:")
    print(f"- Correlación PC1-Posición: {analysis_results['corr_position']:.4f}")
    print(f"- Correlación PC2-Momento: {analysis_results['corr_momentum']:.4f}")
    print(f"- Varianza Explicada (2 PC): {np.sum(analysis_results['explained_variance'][:2]):.2%}")
    print(f"\nRendimiento Modelo Clásico:")
    print(f"- R² Amplitud: {amp_score:.4f}")
    print(f"- R² Fase: {phase_score:.4f}")
    
    # Guardar resultados en un archivo
    with open("results/experiment_summary.txt", "w") as f:
        f.write("RESUMEN DEL EXPERIMENTO W1: LA JAULA CUÁNTICA\n")
        f.write("="*50 + "\n\n")
        f.write(f"Estado de la 'Jaula': {analysis_results['cage_status']}\n\n")
        f.write("Métricas de Calidad:\n")
        f.write(f"- Correlación PC1-Posición: {analysis_results['corr_position']:.4f}\n")
        f.write(f"- Correlación PC2-Momento: {analysis_results['corr_momentum']:.4f}\n")
        f.write(f"- Varianza Explicada (2 PC): {np.sum(analysis_results['explained_variance'][:2]):.2%}\n\n")
        f.write("Rendimiento Modelo Clásico:\n")
        f.write(f"- R² Amplitud: {amp_score:.4f}\n")
        f.write(f"- R² Fase: {phase_score:.4f}\n")
    
    print("\n¡Experimento completado! Los resultados se han guardado en la carpeta 'results/'.")


if __name__ == "__main__":
    run_experiment()
