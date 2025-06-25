# From Big Data to Small Scales: Machine Learning Enhances Microclimate Model Predictions

_Alon Itzkovitch, Idan Sulami, Ronny Doron Efroni, Moni Shahar, and
Ofir Levy_

### Please contact Ofir Levy (levyofir@tauex.tau.ac.il) about the code or data

## Abstract:

1) Microclimates are critical for understanding how organisms interact with their environments, influencing behaviour, physiology, and species distributions. However, traditional physical heat-balance models for predicting ground temperatures in microhabitats often exhibit biases due to unaccounted environmental complexities and poorly constrained parameters. These limitations can hinder ecological research and conservation planning, particularly in the context of climate change.

2) In this study, we demonstrate how high-resolution drone-based mapping and machine learning can improve the accuracy of microclimate models. Using drone imagery, we generated detailed environmental maps, including solar radiation, vegetation indices, and skyview factors, to parameterize a physical heat-balance model. Validation with thermal maps derived from drone-mounted infrared cameras revealed systematic errors in the physical model’s predictions, including over- and underestimations under specific environmental conditions. To address these errors, we applied a random forest machine learning model to predict and correct biases in new prediction maps. 

3) Our results show that machine learning reduced mean absolute errors by over 30% and mean square errors by 50%, while consistently narrowing the range of prediction inaccuracies. Key factors driving biases, such as vegetation cover, solar radiation, and height above ground, were identified, offering valuable insights for improving physical models. The machine learning corrections not only improved accuracy but also highlighted parameters and processes that were previously underrepresented or oversimplified in traditional models.

4) These findings illustrate the potential of combining machine learning with physical modelling to enhance microclimate predictions. This approach provides ecologists and conservation practitioners with a powerful tool to generate accurate, fine-scale microclimate maps, enabling better understanding of species responses to climate change and informing climate-resilient habitat management and conservation strategies. 


# **Repository Directory**:
## See the `Example data` subdirectory for data and metadata. <!-- [link](https://github.com/levyofi/Itzkovitch_el_al_Proceedings_B/tree/main/Example%20data). -->

## See the `Code` subdirectory for codes.  <!--: [link](https://github.com/levyofi/Itzkovitch_el_al_Proceedings_B/tree/main/Codes).-->
