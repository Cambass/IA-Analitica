# 📷 Image Denoising & Anomaly Removal Algorithms

## 📌 Project Overview
This project implements and compares different **Computer Vision filtering techniques** to remove common digital image anomalies (noise). It addresses three specific types of noise:
1.  **Salt & Pepper Noise:** Corrected via Median Filters.
2.  **Gaussian Noise:** Corrected via Gaussian Smoothing.
3.  **Poisson Noise:** Corrected via Bilateral Filters.

A key feature of this repository is the **manual implementation** of convolution kernels using NumPy, compared against optimized implementations from `scikit-image`. This demonstrates a deep understanding of the mathematical operations behind image processing.

*Developed as part of the Artificial Intelligence Specialization at UNIR.*

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Core Libraries:** NumPy (Matrix manipulation), Scikit-image (Image processing), Matplotlib (Visualization), SciPy.
* **Metrics:** PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index).

## 🔬 Methodology

### 1. Noise Simulation
We artificially injected noise into base images to test robustness:
* `random_noise(mode='s&p')`
* `random_noise(mode='gaussian')`

### 2. Filter Implementation (Manual vs. Library)
We implemented filters from scratch to understand kernel convolution:
* **Manual Median Filter:** Iterates over the pixel matrix calculating the median of neighbors.
* **Manual Gaussian Filter:** Applies a weighted kernel to smooth transitions.
* **Comparison:** These were benchmarked against `skimage.filters.median` and `skimage.filters.gaussian`.

### 3. Quantitative Evaluation
Visual inspection is subjective. We used industry-standard metrics to measure restoration quality:
* **PSNR:** Higher values indicate better quality reconstruction.
* **SSIM:** Measures perceived structural similarity (closer to 1.0 is better).

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/camilo-ferro/Computer-Vision-Image-Denoising-Filters.git](https://github.com/camilo-ferro/Computer-Vision-Image-Denoising-Filters.git)
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the analysis:**
    You can explore the step-by-step process in the Jupyter Notebook:
    ```bash
    jupyter notebook notebooks/anomaly_removal_lab.ipynb
    ```

## 📊 Key Findings
* **Median Filter** proved superior for *Salt & Pepper* noise, preserving edges better than Gaussian smoothing.
* **Manual Implementation** achieved similar visual results to libraries but highlighted the computational efficiency importance of optimized C-based libraries like `scikit-image` for production environments.

---
**Author:** Camilo Ferro
*Specialization in Artificial Intelligence - UNIR*
