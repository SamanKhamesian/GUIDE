# GUIDE

**GUIDE: Reinforcement Learning for Behavioral Action Support in Type 1 Diabetes**

Type 1 diabetes (T1D) management requires continuous adjustment of insulin and lifestyle behaviors to maintain blood glucose within a safe target range. Although automated insulin delivery (AID) systems have improved glycemic outcomes, many patients still fail to achieve recommended clinical targets. Current reinforcement learning (RL)-based methods focus primarily on insulin-only treatment and do not provide behavioral recommendations for glucose control. To address this gap, we propose GUIDE, an RL-based decision-support framework designed to complement AID technologies by providing structured behavioral recommendations defined by intervention type, magnitude, and timing, including bolus insulin administration and carbohydrate intake events. GUIDE integrates a patient-specific glucose predictor trained on real-world continuous glucose monitoring data and supports offline and online RL algorithms within a unified environment. The algorithms are evaluated across 25 individuals from the AZT1D dataset and 12 individuals from the OhioT1DM dataset. Among the evaluated algorithms, CQL-BC achieved the best performance on both datasets, with mean time-in-range values of 84.18 $\pm$ 19.89% on AZT1D and 77.64 $\pm$ 8.81% on OhioT1DM. It also maintained time-below-range values of 0.43 $\pm$ 1.27% and 2.81 $\pm$ 3.56%, respectively. Behavioral analysis yielded mean cosine similarities of 0.767 $\pm$ 0.128 on AZT1D and 0.774 $\pm$ 0.162 on OhioT1DM, indicating that the learned policy preserves key structural characteristics of patient action patterns. These findings demonstrate the potential of conservative offline RL with a structured behavioral action space to provide personalized and behaviorally plausible decision support for diabetes management.

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
@misc{khamesian2026guidereinforcementlearningbehavioral,
      title={GUIDE: Reinforcement Learning for Behavioral Action Support in Type 1 Diabetes}, 
      author={Saman Khamesian and Sri Harini Balaji and Di Yang Shi and Stephanie M. Carpenter and Daniel E. Rivera and W. Bradley Knox and Peter Stone and Hassan Ghasemzadeh},
      year={2026},
      eprint={2604.00385},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.00385}, 
}
```
## License

- Released under the [ASU Non-Commercial Research License](LICENSE).
---
