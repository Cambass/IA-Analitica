# 🖼️ Image Segmentation Techniques Comparison

## 📌 Project Overview
This project evaluates and compares three distinct image segmentation techniques to isolate objects of interest from complex backgrounds. The study focuses on processing images with varying lighting conditions and textures (e.g., animals, instruments).

The implemented methods range from classical thresholding to unsupervised machine learning:
1.  **Otsu's Thresholding:** Automatic binary segmentation based on histogram bimodal distribution.
2.  **K-Means Clustering:** Unsupervised learning to group pixels by color similarity ($k=3$).
3.  **Histogram-based Segmentation:** Manual partitioning based on intensity intervals.

*Developed as part of the Computer Perception module - Artificial Intelligence Specialization at UNIR.*

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Computer Vision:** OpenCV (`cv2`)
* **Machine Learning:** Scikit-learn (`KMeans`)
* **Data Manipulation:** NumPy, Matplotlib

## 🔬 Methodology & Results

### 1. Otsu's Method (Global Thresholding)
* **Mechanism:** Calculates the optimal threshold that minimizes intra-class variance.
* **Result:** Successfully created a binary mask (silhouette), clearly separating the subject from the background.
* **Limitation:** Lost internal details (texture/color) of the objects.

### 2. K-Means Clustering (Color Quantization)
* **Mechanism:** Flattens the image into a pixel array and groups them into $k$ clusters based on RGB distance.
* **Result:** Achieved the best balance, simplifying the color palette while preserving the structural integrity of the objects (dogs and guitar). It offered a more semantic segmentation than Otsu.

### 3. Histogram Segmentation
* **Mechanism:** Splits pixel intensities into fixed bins (e.g., 5 intervals).
* **Result:** Produced a fragmented output. While useful for analyzing light distribution, it was less effective for object definition compared to K-Means.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/camilo-ferro/Image-Segmentation-Algorithms-Analysis.git](https://github.com/camilo-ferro/Image-Segmentation-Algorithms-Analysis.git)
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook notebooks/segmentation_analysis.ipynb
    ```

## 📂 Project Structure
* `input_images/`: Original dataset used for testing.
* `notebooks/`: Step-by-step implementation and visualization.

---
**Author:** Camilo Ferro
*Specialization in Artificial Intelligence - UNIR*
