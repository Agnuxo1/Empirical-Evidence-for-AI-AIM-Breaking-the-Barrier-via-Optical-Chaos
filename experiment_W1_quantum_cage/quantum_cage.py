"""
Quantum Cage Experiment - Implementation of quantum simulator and models

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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import expm
from scipy.fft import fft, ifft, fftshift
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from tqdm import tqdm

class QuantumDoubleWell:
    """Simulador cuántico de una partícula en un pozo doble."""
    
    def __init__(self, n_points=128, width=10.0, depth=1.0, barrier=1.0, hbar=1.0, mass=1.0):
        self.n_points = n_points
        self.width = width
        self.depth = depth
        self.barrier = barrier
        self.hbar = hbar
        self.mass = mass
        
        # Espacio de configuración
        self.x = np.linspace(-width/2, width/2, n_points)
        self.dx = self.x[1] - self.x[0]
        
        # Potencial en forma de pozo doble
        self.V = self._double_well_potential()
        
        # Operador Hamiltoniano
        self.H = self._build_hamiltonian()
        
    def _double_well_potential(self):
        """Crea un potencial en forma de pozo doble."""
        x = self.x
        V = -self.depth * (np.exp(-(x - self.width/4)**2 / (2*self.barrier**2)) + 
                          np.exp(-(x + self.width/4)**2 / (2*self.barrier**2)))
        return V
    
    def _build_hamiltonian(self):
        """Construye el operador Hamiltoniano en la base de posiciones."""
        # Término de energía cinética (segunda derivada)
        dx = self.dx
        n = self.n_points
        
        # Matriz de energía cinética (operador laplaciano)
        K = (-2 * np.diag(np.ones(n)) + 
             np.diag(np.ones(n-1), 1) + 
             np.diag(np.ones(n-1), -1))
        K = -self.hbar**2 / (2 * self.mass * dx**2) * K
        
        # Término de energía potencial
        V = np.diag(self.V)
        
        # Hamiltoniano total
        H = K + V
        return H
    
    def evolve_state(self, psi0, t):
        """Evoluciona el estado cuántico en el tiempo."""
        # Operador de evolución temporal
        U = expm(-1j * self.H * t / self.hbar)
        # Aplicar la evolución al estado inicial
        psi_t = U @ psi0
        return psi_t
    
    def generate_trajectory(self, psi0, t_max, n_steps=100):
        """Genera una trayectoria temporal del estado cuántico."""
        t_points = np.linspace(0, t_max, n_steps)
        trajectory = np.zeros((n_steps, self.n_points), dtype=complex)
        
        for i, t in enumerate(t_points):
            trajectory[i] = self.evolve_state(psi0, t)
            
        return trajectory, t_points
    
    def measure_observables(self, psi):
        """Calcula los valores esperados de los observables."""
        # Normalizar el estado
        psi = psi / np.sqrt(np.trapz(np.abs(psi)**2, dx=self.dx))
        
        # Densidad de probabilidad
        prob_density = np.abs(psi)**2
        
        # Valor esperado de la posición
        x_exp = np.trapz(self.x * prob_density, dx=self.dx)
        
        # Incertidumbre en la posición
        x2_exp = np.trapz(self.x**2 * prob_density, dx=self.dx)
        delta_x = np.sqrt(x2_exp - x_exp**2)
        
        # Momento (a través de la transformada de Fourier)
        psi_k = fftshift(fft(psi))
        k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(self.n_points, d=self.dx))
        prob_momentum = np.abs(psi_k)**2
        prob_momentum = prob_momentum / np.trapz(prob_momentum, x=k)
        
        # Valor esperado del momento
        p_exp = np.trapz(k * prob_momentum, x=k)
        
        # Incertidumbre en el momento
        p2_exp = np.trapz(k**2 * prob_momentum, x=k)
        delta_p = np.sqrt(p2_exp - p_exp**2)
        
        return {
            'position': x_exp,
            'momentum': p_exp,
            'delta_x': delta_x,
            'delta_p': delta_p,
            'energy': np.trapz(np.conj(psi) * (self.H @ psi), dx=self.dx).real
        }


class QuantumCageModel(nn.Module):
    """Modelo de red neuronal para predecir la evolución cuántica."""
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=3):
        super(QuantumCageModel, self).__init__()
        
        # Convertir la entrada compleja a real (apilando parte real e imaginaria)
        self.input_processor = nn.Linear(input_dim * 2, hidden_dim)
        
        # Capas ocultas
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.ReLU())
        
        # Capa de salida (parte real e imaginaria)
        self.output_layer = nn.Linear(hidden_dim, input_dim * 2)
        self.activation = nn.ReLU()
        self.input_dim = input_dim
        
    def forward(self, x):
        # x es un tensor complejo, lo convertimos a real apilando parte real e imaginaria
        x_real = x.real
        x_imag = x.imag
        x_combined = torch.cat([x_real, x_imag], dim=-1)
        
        # Procesar la entrada
        x = self.input_processor(x_combined)
        x = self.activation(x)
        
        # Capas ocultas
        for layer in self.hidden_layers:
            x = layer(x)
            
        # Capa de salida
        output = self.output_layer(x)
        
        # Separar parte real e imaginaria
        real = output[:, :self.input_dim]
        imag = output[:, self.input_dim:]
        
        # Combinar en un número complejo
        return torch.complex(real, imag)
    
    def predict_step(self, x):
        """Predice el siguiente paso en la evolución temporal."""
        with torch.no_grad():
            return self(x)


def train_model(model, train_data, val_data, epochs=1000, lr=1e-3):
    """Entrena el modelo con los datos de entrenamiento."""
    # Usamos MSELoss para parte real e imaginaria por separado
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    # Convertir datos de entrenamiento a tensores una sola vez
    X_train_np, y_train_np = train_data
    X_train = torch.tensor(X_train_np, dtype=torch.complex64)
    y_train = torch.tensor(y_train_np, dtype=torch.complex64)
    
    # Convertir datos de validación a tensores una sola vez
    X_val_np, y_val_np = val_data
    X_val = torch.tensor(X_val_np, dtype=torch.complex64)
    y_val = torch.tensor(y_val_np, dtype=torch.complex64)
    
    # Crear DataLoader para el entrenamiento
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # Modo de entrenamiento
        model.train()
        epoch_loss = 0.0
        
        # Entrenamiento por lotes
        for X_batch, y_batch in train_loader:
            # Mover a GPU si está disponible
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            model = model.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            
            # Calcular pérdida para parte real e imaginaria
            loss_real = criterion(outputs.real, y_batch.real)
            loss_imag = criterion(outputs.imag, y_batch.imag)
            loss = loss_real + loss_imag
            
            # Backward y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Guardar pérdida promedio de la época
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validación
        model.eval()
        with torch.no_grad():
            # Mover a GPU si está disponible
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            
            val_outputs = model(X_val)
            val_loss_real = criterion(val_outputs.real, y_val.real)
            val_loss_imag = criterion(val_outputs.imag, y_val.imag)
            val_loss = val_loss_real + val_loss_imag
            val_losses.append(val_loss.item())
            
            # Imprimir progreso cada 10 épocas
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train Loss: {train_losses[-1]:.6f}, "
                      f"Val Loss: {val_losses[-1]:.6f}")
    
    return model, train_losses, val_losses


def analyze_representations(model, data):
    """Analiza las representaciones internas del modelo."""
    # Extraer activaciones de la capa oculta
    activations = []
    
    # Función hook para capturar las activaciones
    def hook(module, input, output):
        activations.append(output.detach().cpu().numpy())
    
    # Registrar hook en la última capa oculta
    # Usamos el último módulo de la lista hidden_layers
    handle = model.hidden_layers[-2].register_forward_hook(hook)
    
    # Mover el modelo a la CPU para evitar problemas de memoria
    device = torch.device("cpu")
    model = model.to(device)
    
    # Convertir datos a tensor y mover a la CPU
    data_tensor = torch.tensor(data, dtype=torch.complex64).to(device)
    
    # Pasar datos por el modelo
    with torch.no_grad():
        # Crear un DataLoader para procesar los datos en lotes
        batch_size = 32
        dataset = torch.utils.data.TensorDataset(data_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Procesar en lotes para evitar problemas de memoria
        for batch in loader:
            _ = model(batch[0])
    
    # Eliminar el hook
    handle.remove()
    
    # Concatenar activaciones
    if activations:
        activations = np.concatenate(activations, axis=0)
    else:
        # Si no se capturaron activaciones, usar la salida del modelo
        with torch.no_grad():
            outputs = model(data_tensor)
            activations = outputs.cpu().numpy()
    
    # Aplicar PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(activations)
    
    return pca_result, pca


def plot_results(trajectory, x, t, title="Evolución Temporal"):
    """Visualiza la evolución temporal de la función de onda."""
    plt.figure(figsize=(10, 6))
    
    # Densidad de probabilidad
    prob_density = np.abs(trajectory)**2
    
    # Normalizar para la visualización
    prob_density = prob_density / np.max(prob_density, axis=1, keepdims=True)
    
    # Crear malla para el gráfico
    T, X = np.meshgrid(t, x)
    
    # Graficar
    plt.pcolormesh(X, T, prob_density.T, shading='auto', cmap='viridis')
    plt.colorbar(label='Densidad de Probabilidad (normalizada)')
    plt.xlabel('Posición (x)')
    plt.ylabel('Tiempo (t)')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_potential(x, V, title="Potencial"):
    """Visualiza el potencial del sistema."""
    plt.figure(figsize=(10, 4))
    plt.plot(x, V, 'r-', linewidth=2)
    plt.xlabel('Posición (x)')
    plt.ylabel('Energía Potencial')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_observables(observables, t, title="Observables"):
    """Visualiza los observables en función del tiempo."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Posición esperada
    axes[0, 0].plot(t, [obs['position'] for obs in observables])
    axes[0, 0].set_xlabel('Tiempo')
    axes[0, 0].set_ylabel('<x>')
    axes[0, 0].set_title('Posición Esperada')
    axes[0, 0].grid(True)
    
    # Momento esperado
    axes[0, 1].plot(t, [obs['momentum'] for obs in observables])
    axes[0, 1].set_xlabel('Tiempo')
    axes[0, 1].set_ylabel('<p>')
    axes[0, 1].set_title('Momento Esperado')
    axes[0, 1].grid(True)
    
    # Energía
    axes[1, 0].plot(t, [obs['energy'] for obs in observables])
    axes[1, 0].set_xlabel('Tiempo')
    axes[1, 0].set_ylabel('Energía')
    axes[1, 0].set_title('Energía Total')
    axes[1, 0].grid(True)
    
    # Relación de incertidumbre
    delta_x = [obs['delta_x'] for obs in observables]
    delta_p = [obs['delta_p'] for obs in observables]
    uncertainty = [dx * dp for dx, dp in zip(delta_x, delta_p)]
    
    axes[1, 1].plot(t, uncertainty, label='Δx·Δp')
    axes[1, 1].axhline(y=0.5, color='r', linestyle='--', label='Límite de Heisenberg')
    axes[1, 1].set_xlabel('Tiempo')
    axes[1, 1].set_ylabel('Δx·Δp')
    axes[1, 1].set_title('Relación de Incertidumbre')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()


def plot_losses(train_losses, val_losses, title="Pérdidas de Entrenamiento"):
    """Visualiza las curvas de pérdida."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Entrenamiento')
    plt.plot(val_losses, label='Validación')
    plt.yscale('log')
    plt.xlabel('Época')
    plt.ylabel('Pérdida (MSE)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Configuración del experimento
    n_points = 128
    width = 10.0
    depth = 1.0
    barrier = 0.5
    t_max = 5.0
    n_steps = 100
    
    # Crear el simulador cuántico
    qm = QuantumDoubleWell(n_points=n_points, width=width, depth=depth, barrier=barrier)
    
    # Estado inicial: paquete de onda gaussiano centrado en x=0
    x0 = 0.0
    sigma = 0.5
    psi0 = np.exp(-(qm.x - x0)**2 / (4 * sigma**2))
    psi0 = psi0 / np.sqrt(np.trapz(np.abs(psi0)**2, dx=qm.dx))  # Normalizar
    
    # Generar trayectoria temporal
    trajectory, t_points = qm.generate_trajectory(psi0, t_max, n_steps)
    
    # Calcular observables
    observables = [qm.measure_observables(psi) for psi in trajectory]
    
    # Visualizaciones
    plot_potential(qm.x, qm.V, "Potencial de Pozo Doble")
    plot_results(trajectory, qm.x, t_points, "Evolución Temporal de la Función de Onda")
    plot_observables(observables, t_points, "Evolución de los Observables")
    
    print("Simulación cuántica completada. Los resultados se han guardado en las figuras.")
