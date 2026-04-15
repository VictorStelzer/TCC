# Core

Sistema de visão computacional profissional focado em inferência on-device para auxílio na navegação.

## Estrutura do Projeto

* `data/`: Dados brutos e processados para o treinamento do YOLOv8.
* `models/`: Armazena os pesos do modelo exportados (`.pt`, `.tflite`).
* `src/`: Core do projeto (código-fonte principal).
  * `core_initializer.py`: Validação se o hardware (GPU/CPU) está pronto.
  * `trainer_engine.py`: Boilerplate para carregamento e treinamento de modelos (YOLOv8 Nano).
  * `stream_validator.py`: Validação de rotinas do OpenCV e fluxo de captura.
* `scripts/`: Utilitários gerais e scripts de conversão on-device.
* `environment_config.yaml`: Metadados e classes mapeadas (pedestre, veiculo, entre outros).
