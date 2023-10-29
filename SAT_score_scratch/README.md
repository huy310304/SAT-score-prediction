## Initial Model Exploration

The initial phase of the project involved training a neural network model from scratch using NumPy for SAT score prediction. The results and findings from this initial exploration phase are detailed below.

<img src="../plotting_results/sat_regression_frame_plot.png" width="600"/>

### Training and Results

- **Training Process:** The initial model was trained using a dataset in the [original csv file](../data/GPA_Small.csv) that included SAT scores and related input features in `data`, with GPA being a primary focus.

- **Promising Potential:** The results of this initial training phase were highly promising. The model exhibited significant potential, achieving an accuracy of approximately 65% after 30 epochs of training. This marked the foundation upon which subsequent optimizations and enhancements were built.

- **Regression Focus:** The primary objective of this initial model was to establish a robust regression model capable of accurately predicting SAT scores based on various input factors, with GPA being a key predictor.

### Future Directions

The success of this initial model has paved the way for further refinements and enhancements. Future steps in this project will involve:

- **Hyperparameter Tuning:** Fine-tuning the model's hyperparameters to optimize its performance, potentially leading to higher accuracy and faster convergence.

- **Data Preprocessing:** Refining data preprocessing techniques to handle missing values, scale input features, and conduct feature engineering as needed.

- **Loss Function Evaluation:** Experimenting with different loss functions to identify the one that best suits the regression task.

- **Visualization:** Continued monitoring of the model's training progress and outcomes through visualizations to gain insights into its learning process.

- **Extension to Classification:** The project will be extended to include a classification aspect to predict college admission outcomes based on the academic profile.

This initial exploration phase has set the stage for a more comprehensive and accurate SAT score prediction model, with the ultimate aim of providing valuable insights and predictions to students, educators, and institutions.
