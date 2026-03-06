# GUIDE

**GUIDE: Reinforcement Learning for Behavioral Action Support in Type 1 Diabetes**

Type 1 Diabetes (T1D) management requires continuous adjustment of insulin and lifestyle behaviors to maintain blood glucose within a safe target range. 
Although automated insulin delivery (AID) systems have improved glycemic outcomes, many patients still fail to achieve recommended clinical targets, warranting new approaches to improve glucose control in patients with T1D. 
While reinforcement learning (RL) has been utilized as a promising approach, currently RL-based methods focus primarily on insulin-only treatment and do not provide behavioral recommendations for glucose control. 
To address this gap, we propose GUIDE, an RL-based decision-support framework designed to complement AID technologies by providing behavioral action recommendations for just-in-time prevention of abnormal glucose events. 
GUIDE generates structured actions that jointly model action type, magnitude, and timing, including bolus insulin administration and carbohydrate intake actions. 
GUIDE integrates a patient-specific glucose level predictor trained on real-world continuous glucose monitoring data and supports both offline and online RL algorithms within a unified environment. 
We evaluate off-policy and on-policy methods across 25 individuals with T1D using standardized glycemic metrics. 
Among the evaluated approaches, Conservative Q-Learning with Behavior Cloning (CQL-BC) demonstrates the highest average time-in-range, reaching 85.49% while maintaining low hypoglycemia exposure. 
Behavioral similarity analysis further indicates that the learned CQL-BC policy preserves key structural characteristics of patient action patterns, achieving a mean cosine similarity of 0.87 ± 0.09 across subjects. 
These findings suggest that conservative offline RL with a structured behavioral action space can provide clinically meaningful and behaviorally plausible decision support for personalized diabetes management.

---

## 📁 Dataset Setup

### AZT1D Dataset

- **Download from Mendeley:**  
  https://data.mendeley.com/datasets/gk9m674wcx/1

- **Directory structure:**  
  Place ```AZT1D``` folder in the ```./GUIDE/dataset/``` directory:

---

## ⚙️ Environment Setup

- **Python version:** `3.12`

- **Install dependencies:**

  Create a virtual environment (optional but recommended):

  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

  Then install required packages:

  ```bash
  pip install -r requirements.txt
  ```

  `requirements.txt` includes:

  ```
  matplotlib==3.10.8
  numpy==2.4.2
  pandas==3.0.1
  scikit_learn==1.8.0
  scipy==1.17.1
  statsmodels==0.14.6
  tensorflow==2.20.0
  torch==2.7.0
  ```
---

## 📖 Citation

If you use GLIMMER in your work, please cite:

```bibtex
@article{khamesian2026guide}
```

---
