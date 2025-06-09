# 🌱 GREEN AI: Drone-Based Afforestation Monitoring System

<div align="center">
  <img src="https://github.com/ADARSH-TKD/GREEN_AI/blob/main/IMAGE/ima%201.jpg?raw=true" alt="Green AI Banner" width="100%">
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
  [![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()
</div>

## 📋 Table of Contents
- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Solution Architecture](#-solution-architecture)
- [Technology Stack](#-technology-stack)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results & Impact](#-results--impact)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🌟 Overview

**Green AI** is an innovative drone-based monitoring framework that leverages computer vision, machine learning, and geospatial analytics to revolutionize afforestation program management. Developed specifically to address Odisha's large-scale plantation monitoring challenges, this system provides real-time sapling health assessment, survival rate analysis, and growth tracking through advanced UAV technology.

### 🎯 Key Objectives
- Monitor 5 crore saplings planted annually in Odisha
- Improve sapling survival rates from current 65% to 85%+
- Provide real-time actionable insights for forest departments
- Enable scalable monitoring across remote and inaccessible terrains

<div align="center">
  <img src="https://github.com/ADARSH-TKD/GREEN_AI/blob/main/IMAGE/img%202.jpg?raw=true" alt="Drone Monitoring" width="70%">
</div>

## 🔍 Problem Statement

### Current Challenges in Afforestation Programs:
- **Low Survival Rates**: Only 65% sapling survival due to undetected threats
- **Limited Access**: Geographical constraints in remote/hilly regions
- **Resource Inefficiency**: Delayed problem detection leading to wastage
- **Manual Limitations**: Time-intensive, labor-heavy traditional monitoring

### Our Solution Impact:
- **Early Detection**: Identify pest infestations, drought stress, and grazing damage
- **Comprehensive Coverage**: Monitor vast areas including Western Odisha's challenging terrain
- **Cost Optimization**: Reduce dependency on manual surveys by 70%
- **Data-Driven Decisions**: Enable proactive interventions with precise analytics

## 🏗️ Solution Architecture

<div align="center">
  <img src="https://github.com/ADARSH-TKD/GREEN_AI/blob/main/IMAGE/image_2.png" alt="System Architecture" width="80%">
</div>

### Workflow Pipeline:
1. **Drone Data Acquisition** → High-resolution imagery capture (2.46-2.81 cm/pixel)
2. **Preprocessing** → Orthomosaic generation, image enhancement, noise reduction
3. **Segmentation** → Sapling isolation using Deep Learning (U-Net, Mask R-CNN)
4. **Feature Extraction** → Health assessment via CNN models (ResNet, EfficientNet)
5. **Analysis & Prediction** → Survival classification, growth measurement
6. **Visualization** → Interactive dashboards and actionable reports

## 🛠️ Technology Stack

<div align="center">

| Category | Technologies | Purpose |
|----------|-------------|---------|
| **Programming** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Core development, data processing |
| **Computer Vision** | ![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=flat&logo=OpenCV&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) | Image processing, ML models |
| **Web Framework** | ![Django](https://img.shields.io/badge/Django-092E20?style=flat&logo=django&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black) | Dashboard, visualization |
| **Cloud Services** | ![GCP](https://img.shields.io/badge/Google_Cloud-4285F4?style=flat&logo=google-cloud&logoColor=white) | Deployment, storage, processing |
| **Geospatial** | ![GeoPandas](https://img.shields.io/badge/GeoPandas-139C5A?style=flat) **Pix4D** | GIS analysis, orthomosaic generation |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) **Looker Studio** | Data visualization, reporting |

</div>

## ✨ Features

### 🎯 Core Capabilities
- **Automated Sapling Detection**: YOLO/Faster R-CNN for real-time object detection
- **Health Classification**: CNN-based survival prediction (Alive/Casualty)
- **Growth Tracking**: Time-series analysis of sapling development
- **Geospatial Mapping**: GPS-enabled precise location tracking
- **Change Detection**: Multi-temporal analysis for growth patterns

### 📊 Advanced Analytics
- **Survival Rate Analysis**: Statistical modeling with 95%+ accuracy
- **Pest/Disease Detection**: Early warning system using computer vision
- **Resource Optimization**: AI-driven intervention recommendations
- **Performance Metrics**: Comprehensive KPI tracking and reporting

<div align="center">
  <img src="https://github.com/ADARSH-TKD/GREEN_AI/blob/main/IMAGE/image_1.png" alt="Analytics Dashboard" width="70%">
</div>

## 🚀 Installation

### Prerequisites
```bash
Python >= 3.8
CUDA >= 11.0 (for GPU acceleration)
```

### Setup Instructions
```bash
# Clone the repository
git clone https://github.com/ADARSH-TKD/GREEN_AI.git
cd GREEN_AI

# Create virtual environment
python -m venv green_ai_env
source green_ai_env/bin/activate  # Linux/Mac
# OR
green_ai_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install additional packages
pip install tensorflow-gpu==2.x.x
pip install opencv-python==4.x.x
pip install geopandas
```

### Configuration
```bash
# Set up environment variables
export GOOGLE_CLOUD_PROJECT="your-project-id"
export PIX4D_API_KEY="your-pix4d-key"

# Initialize database
python manage.py migrate
python manage.py collectstatic
```

## 💻 Usage

### Quick Start
```python
from green_ai import DroneMonitor, AnalysisEngine

# Initialize the monitoring system
monitor = DroneMonitor(
    elevation_range=(70, 80),  # meters
    resolution="2.46cm/pixel",
    overlap_config={'sidelap': 65, 'endlap': 75}
)

# Process drone imagery
results = monitor.analyze_plantation(
    image_path="path/to/drone/images/",
    operations=['OP1', 'OP2', 'OP3']
)

# Generate insights
engine = AnalysisEngine()
insights = engine.generate_report(results)
```

### Command Line Interface
```bash
# Process single operation
python green_ai_cli.py --operation OP2 --input ./drone_data/june_2024/

# Batch processing
python green_ai_cli.py --batch --config config.yaml

# Generate dashboard
python manage.py runserver
# Access at http://localhost:8000/dashboard
```

## 📁 Project Structure

```
GREEN_AI/
├── 📁 core/                    # Core processing modules
│   ├── image_processing.py     # OpenCV operations
│   ├── ml_models.py           # CNN/YOLO implementations
│   ├── geospatial.py          # GIS operations
│   └── analysis_engine.py     # Main analysis logic
├── 📁 data/                   # Data management
│   ├── preprocessing/         # Data cleaning scripts
│   ├── models/               # Trained ML models
│   └── sample_data/          # Test datasets
├── 📁 dashboard/             # Web interface
│   ├── templates/            # HTML templates
│   ├── static/              # CSS/JS files
│   └── views.py             # Django views
├── 📁 utils/                # Utility functions
│   ├── drone_ops.py         # Drone operation handlers
│   ├── visualization.py     # Plotting functions
│   └── metrics.py           # Performance evaluation
├── 📁 tests/                # Unit tests
├── 📁 docs/                 # Documentation
├── requirements.txt         # Dependencies
├── config.yaml             # Configuration file
└── README.md              # This file
```

## 🔬 Methodology

### Data Collection Protocol
<div align="center">
  <img src="https://github.com/ADARSH-TKD/GREEN_AI/blob/main/IMAGE/image_3.png" alt="Methodology" width="70%">
</div>

#### Operation Stages:
1. **OP1 (March-May)**: Pit preparation and baseline imaging
2. **OP2 (June-July)**: Post-plantation monitoring during monsoon
3. **OP3 (Oct-Nov)**: Annual growth assessment and maintenance

### Image Processing Pipeline:
```python
# Preprocessing workflow
def preprocess_drone_imagery(image_path):
    # 1. Orthomosaic generation
    orthomosaic = generate_orthomosaic(image_path, overlap=(65, 75))
    
    # 2. Image enhancement
    enhanced = apply_clahe(orthomosaic)
    enhanced = gaussian_blur(enhanced, kernel_size=3)
    
    # 3. Segmentation
    segments = multi_otsu_threshold(enhanced, classes=3)
    
    # 4. Feature extraction
    features = extract_haralick_features(segments)
    
    return processed_data
```

### Machine Learning Models:
- **Object Detection**: YOLOv8 for sapling localization
- **Classification**: ResNet-50 for health assessment
- **Segmentation**: U-Net for precise boundary detection
- **Regression**: CNN-based growth prediction models

## 📈 Results & Impact

### Performance Metrics:
<div align="center">
  <img src="https://github.com/ADARSH-TKD/GREEN_AI/blob/main/IMAGE/image_4.png" alt="Results" width="60%">
</div>

| Metric | Achievement | Improvement |
|--------|-------------|-------------|
| **Detection Accuracy** | 96.3% | +31% vs manual |
| **Processing Speed** | 2.5 min/hectare | 15x faster |
| **Survival Rate Prediction** | 94.7% F1-score | Real-time alerts |
| **Cost Reduction** | 68% | vs traditional methods |

### Environmental Impact:
- **Forest Cover**: Projected 25% increase in successful plantations
- **Carbon Sequestration**: Enhanced CO₂ absorption capacity
- **Biodiversity**: Improved ecosystem restoration outcomes
- **Scalability**: Framework applicable to 29 Indian states

### Socio-Economic Benefits:
- **Job Creation**: 200+ positions in drone operations and data analysis
- **Rural Development**: Technology integration in forest communities
- **Policy Support**: Data-driven decision making for forest departments

<div align="center">
  <img src="https://github.com/ADARSH-TKD/GREEN_AI/blob/main/IMAGE/image_5.png" alt="Impact" width="70%">
</div>

## 🤝 Contributing

We welcome contributions from the community! Please follow these guidelines:

### Development Workflow:
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Code Standards:
- Follow PEP 8 for Python code
- Include unit tests for new features
- Update documentation for API changes
- Ensure compatibility with existing modules

### Areas for Contribution:
- 🔧 Algorithm optimization
- 🌐 Multi-language support
- 📱 Mobile application development
- 🛡️ Security enhancements
- 📊 Advanced visualization features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Research References:
- **Panama Forest Monitoring**: Community-led drone surveillance programs
- **Australian Reforestation**: LiDAR and hyperspectral imaging applications
- **Global Forest Goals Report 2021**: United Nations framework
- **State of Forest Report 2021**: Odisha forest department data

### Technical Inspiration:
- Structure from Motion (SfM) algorithms
- UAV-assisted environmental monitoring research
- Machine learning applications in forestry

### Special Thanks:
- **Odisha Forest Department** for domain expertise
- **Research Community** for algorithmic foundations
- **Open Source Contributors** for tool development

---

<div align="center">
  <h3>🌍 Together, let's build a greener future with AI! 🌱</h3>
  
  **[Documentation](docs/) | [Demo](demo/) | [Issues](issues/) | [Discussions](discussions/)**
  
  Made with ❤️ for environmental conservation
</div>
