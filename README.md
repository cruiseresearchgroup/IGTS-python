> **If you use the resources (algorithm, code and dataset) presented in this repository, please cite our paper.**  
*The BibTeX entry is provided at the bottom of this page. 

# (IGTS) Temporal Segmentation for multivariate time series

Information gain-based metric for recognizing transitions in human activities

This work aims to recognize transition times in multivariate time series focusing on human activities. No generic method has been proposed for extracting transition times at different levels of activity granularity. Existing work in human behavior analysis and activity recognition has mainly used predefined sliding windows or fixed segments, either at low-level, such as standing or walking, or high-level, such as dining or commuting to work. We present an Information Gain-based Temporal Segmentation method (IGTS), an unsupervised segmentation technique, to find the transition times in human activities and daily routines, from heterogeneous sensor data. The proposed IGTS method is applicable for low-level activities, where each segment captures a single activity, such as walking, that is going to be recognized or predicted, and also for high-level activities. The heterogeneity of sensor data is dealt with a data transformation stage. The generic method has been thoroughly evaluated on a variety of labeled and unlabeled activity recognition and routine datasets from smartphones and device-free infrastructures. The experiment results demonstrate the robustness of the method, as all segments of low- and high-level activities can be captured from different datasets with minimum error and high computational efficiency.

This repository contains resources developed within the following paper:

	Sadri, A., Ren, Y., & Salim, F. D. (2017). Information gain-based metric for recognizing transitions in 
	human activities. Pervasive and Mobile Computing, 38, 92-109.
  
You can find the [paper](https://github.com/cruiseresearchgroup/IGTS-python/blob/master/paper/1-s2.0-S1574119217300081-main.pdf) and [presentation](https://github.com/cruiseresearchgroup/IGTS-python/blob/master/Presentation/IGTS-%20Presentation.pptx) in this repository. 

Alternative link: https://www.sciencedirect.com/science/article/pii/S1574119217300081

## Contents of the repository
This repository contains resources used and described in the paper.

The repository is structured as follows:
- `paper/`: Formal description of the algorithm and evaluation results
- `code/`: The python project code
- `presentation/`: The presentation slides

## Code
For `matlab` code, please refer to this. The python codes are available in `code/`. 
	- `IGTS.py` includes following functions:
		- "Clean_TS": This function normalizes the time series and doubles the number of the time series to address the hetergenousity in the time series.
		- "DP_IGTS": Implementation of the dynamic programming for IGTS
		- "TopDown_IGTS": Implementation of the TOP-Down algorithm for IGTS
		- "IG_Cal": It calculates the information gain
		- "Sh_Entropy": It calculates the entropy
  - `demo_syntheticdata.py` is a demo script. 

The input is m time series with the length of n that should be stored in an m*n matrix.

## Possible Applications

## Citation
If you use the resources presented in this repository, please cite (using the following BibTeX entry):
```
@article{sadri2017information,
  title={Information gain-based metric for recognizing transitions in human activities},
  author={Sadri, Amin and Ren, Yongli and Salim, Flora D},
  journal={Pervasive and Mobile Computing},
  volume={38},
  pages={92--109},
  year={2017},
  publisher={Elsevier}
}
```
