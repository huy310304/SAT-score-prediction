# Optimized Model Achievement

Building upon the insights gained from the initial model, an iterative process of fine-tuning and optimization was embarked upon. By meticulously applying advanced activation functions, dropout regularization, and learning rate reduction strategies, the model's accuracy and convergence were substantially improved. The optimized model's performance is demonstrated below:

<img src="plotting_results/sat_regression_final_plot.png" width="600"/>

Notably, the optimized model achieved a remarkable accuracy of 90%, showcasing the efficacy of the implemented techniques. Moreover, the convergence time was significantly reduced to merely 5-10 epochs, signifying the efficiency of the training process.

It's important to acknowledge that while individual epoch times were extended due to increased complexity, this aspect can be modulated by adjusting the batch size. Increasing the batch size offers an avenue to expedite convergence while maintaining the model's enhanced accuracy.

# Additional Data Preprocessing Techniques

Beyond the model architecture, data preprocessing also played a pivotal role in refining prediction accuracy. Two distinct techniques were integrated into the pipeline to further enhance results:

## Binary Splitting Technique for `hsize` and `hsrank`

The binary splitting technique was employed to encode the `hsize` and `hsrank` features as binary representations. This technique enhances the model's ability to capture nuanced relationships within these categorical variables, contributing to the overall prediction accuracy.

<img src="plotting_results/sat_regression_binaryspliting_plot.png" width="600"/>

## PCA (Principal Component Analysis) Technique for Feature Reduction

Principal Component Analysis (PCA) was utilized to reduce the feature space to two dimensions. This technique enables a compact representation of the data while retaining its essential variance. The resulting reduction in feature dimensions contributes to a streamlined and efficient training process.

<img src="plotting_results/sat_regression_binary_pca.png" width="600"/>

Both binary splitting and PCA exhibited comparable accuracy and loss trends over time when contrasted against the final optimized model. This consistency underscores the robustness of the model's predictions and the efficacy of the chosen preprocessing strategies.

The culmination of these efforts and techniques underscores the advancement achieved in predicting SAT scores. By diligently refining the model architecture, leveraging sophisticated data preprocessing, and implementing optimization strategies, the project attains a level of accuracy and efficiency that is poised to yield meaningful insights in educational evaluation and assessment.
