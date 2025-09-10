# **Land Cover Monitor System**

This repository contains all the scripts and resources used in the development of the **Land Cover Monitor System**, including both the **model training pipeline** and the **user interface application**.

---

## **Repository Structure**

- **`User Interface - Landcover-Monitor/`**  
  Contains the fully developed application, ready for deployment.  
  For detailed deployment instructions, please refer to **Appendix D** of the main report.

- **`Scripts/`**  
  Includes various scripts used for **model training**, **evaluation**, and **data processing**.

- **`Model/`**  
  Contains a selection of **trained model files** together with their corresponding **training loss curves** saved in `.csv` format.

- **`Satellite Images/`**  
  Satellite images are provided for **demonstration** and **testing purposes**.

---

## **Main Jupyter Notebooks**

- **`Comp702_Modular.ipynb`**  
  The main notebook that integrates and executes the modular scripts for **training the deep learning model**.

- **`Comp702_Initial_Tests.ipynb`**  
  Contains the **initial data exploration** and **prototype testing** carried out **before developing the modular scripts**.

- **`Comp702_EvaluatingResults.ipynb`**  
  Used for **evaluating model performance**, visualising metrics, and comparing results.

- **`COMP702_GettingSatelliteIMG.ipynb`**  
  A utility notebook used to **download and manage satellite imagery** for training and evaluation.

---

## **Acknowledgement of External Libraries**

This project utilises several open-source Python libraries. I acknowledge and thank the developers and maintainers of the following key packages:

### **Machine Learning & Deep Learning**
- **PyTorch** (`torch`, `torch.nn`, `torch.utils.data`) – Used for building, training, and evaluating deep learning models.
- **TorchVision** (`torchvision`, `torchvision.datasets`, `torchvision.transforms`) – Provides access to computer vision datasets, image transformations, and pre-trained models.

### **Data Handling & Processing**
- **Pandas** (`pandas`) – Used for structured data manipulation and analysis.
- **NumPy** (`numpy`) – Provides support for numerical computations and array-based operations.
- **Pillow** (`PIL.Image`) – Handles image loading and basic image processing.

### **Geospatial Data & Remote Sensing**
- **GeoPandas** (`geopandas`) – Facilitates the handling and analysis of geospatial vector data.
- **GeoJSON** (`geojson`) – Used for working with geographic data in GeoJSON format.
- **Rasterio** (`rasterio`) – Reads, writes, and processes
