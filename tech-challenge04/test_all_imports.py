#!/usr/bin/env python3
"""
Script melhorado para testar importações e mostrar erros reais
Tech Challenge - Análise de Vídeo
"""

import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TESTE DETALHADO DE INSTALAÇÃO DAS BIBLIOTECAS")
print("=" * 80)

# Função para testar cada biblioteca individualmente
def test_import(module_name, import_statement, version_check):
    """Testa importação e mostra erro detalhado se falhar"""
    try:
        exec(import_statement)
        version = eval(version_check) if version_check else "OK"
        print(f"✅ {module_name:<20} {version}")
        return True, None
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e).split('\n')[0]  # Primeira linha do erro
        print(f"❌ {module_name:<20} {error_type}: {error_msg}")
        return False, str(e)

# Definir testes
tests = [
    ("opencv-python", "import cv2", "cv2.__version__"),
    ("mediapipe", "import mediapipe", "mediapipe.__version__"),
    ("deepface", "from deepface import DeepFace", "None"),
    ("ultralytics", "from ultralytics import YOLO; import ultralytics", "ultralytics.__version__"),
    ("numpy", "import numpy", "numpy.__version__"),
    ("pandas", "import pandas", "pandas.__version__"),
    ("matplotlib", "import matplotlib", "matplotlib.__version__"),
    ("seaborn", "import seaborn", "seaborn.__version__"),
    ("tqdm", "import tqdm", "tqdm.__version__"),
    ("scikit-learn", "import sklearn", "sklearn.__version__"),
    ("tensorflow", "import tensorflow", "tensorflow.__version__"),
    ("torch", "import torch", "torch.__version__"),
]

# Executar testes
success_count = 0
failed = []

for module_name, import_stmt, version_check in tests:
    success, error = test_import(module_name, import_stmt, version_check)
    if success:
        success_count += 1
    else:
        failed.append((module_name, error))

# Verificações especiais
print("\n" + "=" * 80)
print("VERIFICAÇÕES DETALHADAS:")
print("-" * 80)

# Python
print(f"\n📌 Python: {sys.version}")

# CUDA/GPU
print("\n🎮 GPU/CUDA:")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA disponível: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("  ⚠️  GPU não detectada - usando versão CPU")
        if "cpu" in torch.__version__:
            print("  💡 Instale versão GPU com:")
            print("     pip uninstall torch torchvision torchaudio -y")
            print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
except Exception as e:
    print(f"  ❌ Erro ao verificar PyTorch: {e}")

# TensorFlow GPU
print("\n🧠 TensorFlow:")
try:
    import tensorflow as tf
    print(f"  TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"  GPUs detectadas: {len(gpus)}")
    
    # Verificar tf_keras
    try:
        import tf_keras
        print("  ✅ tf_keras instalado")
    except:
        print("  ⚠️  tf_keras NÃO instalado - necessário para DeepFace")
        print("     Instale com: pip install tf_keras")
        
except Exception as e:
    print(f"  ❌ Erro ao verificar TensorFlow: {e}")

# Verificar ml_dtypes
print("\n🔧 Dependências críticas:")
try:
    import ml_dtypes
    print(f"  ml_dtypes version: {ml_dtypes.__version__}")
except:
    print("  ❌ ml_dtypes não encontrado")

# Resumo
print("\n" + "=" * 80)
print("RESUMO:")
print("-" * 80)
print(f"✅ Bibliotecas funcionando: {success_count}/{len(tests)}")
print(f"❌ Bibliotecas com problemas: {len(failed)}")

if failed:
    print("\n⚠️  PROBLEMAS ENCONTRADOS:")
    for lib, error in failed:
        print(f"\n{lib}:")
        print(f"  {error[:200]}...")  # Primeiros 200 caracteres do erro

# Soluções sugeridas
print("\n" + "=" * 80)
print("SOLUÇÕES SUGERIDAS:")
print("-" * 80)

solutions = []

# Verificar problemas comuns
for lib, error in failed:
    if "tf_keras" in str(error):
        solutions.append("pip install tf_keras")
    elif "deepface" in lib.lower():
        solutions.append("pip install tf_keras  # Necessário para DeepFace com TF 2.15")

try:
    import torch
    if not torch.cuda.is_available() and "cpu" in torch.__version__:
        solutions.append("\n# Para GPU (CUDA):")
        solutions.append("pip uninstall torch torchvision torchaudio -y")
        solutions.append("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
except:
    pass

if solutions:
    print("Execute os seguintes comandos:")
    for sol in solutions:
        print(f"  {sol}")
else:
    print("✅ Tudo parece estar funcionando!")

print("=" * 80)

# Teste rápido de funcionalidade
if success_count == len(tests):
    print("\n🚀 TESTE DE FUNCIONALIDADE RÁPIDA:")
    print("-" * 80)
    try:
        # Teste OpenCV
        import cv2
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        print("✅ OpenCV: Criação de imagem OK")
        
        # Teste MediaPipe
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        print("✅ MediaPipe: Inicialização OK")
        
        # Teste PyTorch GPU
        import torch
        if torch.cuda.is_available():
            tensor = torch.zeros(1).cuda()
            print("✅ PyTorch: GPU funcionando")
        
        print("\n🎉 AMBIENTE PRONTO PARA O PROJETO!")
        
    except Exception as e:
        print(f"⚠️  Erro no teste funcional: {e}")