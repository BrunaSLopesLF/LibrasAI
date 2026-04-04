import os
import numpy as np

# Configurações iguais ao do pipeline
DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
CLASSES = ["OI", "OBRIGADO", "BOM_DIA", "SIM", "NAO"]
SEQUENCES_PER_CLASS = 30
SEQUENCE_LENGTH = 30
FEATURE_SIZE = 42

def generate_dummy_data():
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    
    for c in CLASSES:
        class_dir = os.path.join(DATA_RAW_DIR, c)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(SEQUENCES_PER_CLASS):
            # Gera dados aleatorios entre 0 e 1 simulando as posições normalizadas
            # Para que as features não sejam idênticas e haja algum padrao de classe
            # daremos um peso (bias) dependendo do índice da classe
            base_noise = np.random.rand(SEQUENCE_LENGTH, FEATURE_SIZE)
            bias = CLASSES.index(c) * 0.1
            dummy_sequence = base_noise + bias
            
            # Formato esperado: (30, 42)
            filepath = os.path.join(class_dir, f"{i}.npy")
            np.save(filepath, dummy_sequence)
            
    print(f"Gerados dados sintéticos em: {DATA_RAW_DIR}")

if __name__ == "__main__":
    generate_dummy_data()
