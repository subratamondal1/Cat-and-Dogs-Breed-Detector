# <center>🐶🐱Cat & Dog Breed Detector🐕🐈</center>
---

## Project Overview
---
This deep learning project presents a sophisticated Cat & Dog breed detector built using the advanced Fastai framework. Our objective was to accurately classify pet images into 37 distinct categories, representing various breeds. To achieve this, we utilized the challenging Oxford-IIIT Pet Dataset, known for its diverse pet images with complex variations in scale, pose, and lighting.

## Fastai Computer Vision Pipeline for Cat & Dog Breed Detection
---

### 1. Data Loading
---
- We focused on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), renowned for its 37 distinct pet categories and high-quality images.
- Essential libraries, including FastAi 2.7.12, were imported to support this computer vision project.
- The FastAi vision package was imported to leverage its modules and functions for computer vision tasks.
- We employed the `get_image_files` function to retrieve image file paths.
- The Oxford-IIIT Pet Dataset was downloaded and untarred from the provided URL.
- A list of image file names within the 'images' directory of the dataset was obtained using the `get_image_files` function.
- We displayed the total image count in the dataset, which stands at 7390 images.
- Additionally, we showcased the file paths of the first 10 images in the dataset, offering insights into the data's structure and location.

### 2. Data Preparation
---
- **Statistical Normalization**: Importing critical statistics from FastAi's `imagenet_stats` for image data normalization.
- **Image Augmentation and Cropping**: Incorporating essential image transformation functions like `aug_transforms` and `RandomResizedCrop` from FastAi to introduce variability and robustness into the dataset.
- **Item Transforms**: Defining `item_tfms` to specify item-level transformations, including random resized crops with dimensions of 460 pixels, enhancing dataset diversity.
- **Batch Transforms**: Creating a list of batch-level transformations, denoted as `batch_tfms`, to apply operations such as resizing to 224 pixels, maximum warping, and data normalization using `imagenet_stats`.

These meticulous data preparation steps ensure that the dataset is appropriately conditioned for subsequent phases of the computer vision pipeline, setting the stage for successful model training, validation, and evaluation.

### 3. Creating Data Loaders
---
- **DataBlock Definition**: Establishing a DataBlock named 'pets' to orchestrate data processing. It defines key aspects, including data blocks (Image and Category), image file acquisition, random data splitting into training and validation sets, and category label extraction from file names using regular expressions.
- **Item-Level and Batch-Level Transformations**: Configuring `item_tfms` and `batch_tfms` within the DataBlock to ensure consistent preprocessing of each image and batch for model readiness.
- **Data Loaders Creation**: Creating data loaders ('dls') using the 'pets' DataBlock for efficient data loading and batching. The training dataset ('dls.train_ds') contains 5912 images, while the validation dataset ('dls.valid_ds') contains 1478 images, both spanning 37 distinct pet breeds. A batch size of 64 was specified for data loaders.
- **Data Set Overview**: Providing a sneak peek into the training and validation datasets by displaying a sample of their elements, consisting of PIL images and corresponding category labels. Additionally, revealing the 37 distinct pet breed classes.

### 4. Defining Learner (Model) & Learning Rate
---
### 4. Defining Learner (Model) & Learning Rate
---
- **Transfer Learning with Pretrained Model**: Leveraging the power of transfer learning, we started with a pretrained model as the foundation of our Cat & Dog breed detector. Specifically, we used a ResNet-50 architecture that had been pretrained on the vast ImageNet dataset. This pretrained model had already learned to recognize a wide range of low-level features, such as edges, textures, and basic shapes, making it a valuable starting point for our specific task.

- **Mixed-Precision Training**: To optimize training efficiency, we employed mixed-precision training techniques using FastAi's `to_fp16()` method. This allowed us to use lower-precision data types during training, reducing memory consumption and speeding up the training process while maintaining model accuracy.

- **Model Architecture**: Our learner (model) was instantiated with key parameters, including data loaders ('dls'), the ResNet-50 architecture ('arch=resnet50'), and the utilization of a pretrained model ('pretrained=True'). This pretrained model served as a feature extractor, preserving the knowledge it had gained from ImageNet.

- **Fine-Tuning the Top Layers**: While the initial layers of the pretrained model were frozen (kept fixed), the top layers, responsible for higher-level feature extraction and classification, were trainable. This allowed our model to adapt its learned features to the specific task of classifying pet breeds.

- **Evaluation Metrics**: To assess the model's performance, we equipped the learner with evaluation metrics, including accuracy and error rate.

- **Learning Rate Finder**: To determine the optimal learning rate for our fine-tuning process, we used the `learn.lr_find()` method. The reported learning rate range, `slice(0.0001, 0.01, None)`, provided a suitable range for adaptive learning rate adjustments during training.

These steps lay the foundation for our Cat & Dog breed detector, incorporating transfer learning to capitalize on the knowledge acquired by the pretrained model and adapt it to the nuances of pet breed classification.


### 5. Training & Saving the Model
---
- Displaying training and validation metrics across 10 epochs:

| Epoch | Train Loss | Valid Loss | Accuracy | Error Rate | Time   |
|-------|------------|------------|----------|------------|--------|
| 0     | 0.666741   | 0.321287   | 0.895129 | 0.104871   | 01:27  |
| 1     | 0.496385   | 0.430292   | 0.875507 | 0.124493   | 01:26  |
| 2     | 0.483526   | 0.561120   | 0.868065 | 0.131935   | 01:27  |
| 3     | 0.375414   | 0.347090   | 0.908660 | 0.091340   | 01:28  |
| 4     | 0.289794   | 0.372382   | 0.899188 | 0.100812   | 01:27  |
| 5     | 0.215737   | 0.319737   | 0.920839 | 0.079161   | 01:25  |
| 6     | 0.156200   | 0.319586   | 0.924899 | 0.075101   | 01:27  |
| 7     | 0.110415   | 0.235808   | 0.936401 | 0.063599   | 01:27  |
| 8     | 0.078930   | 0.260270   | 0.929635 | 0.070365   | 01:27  |
| 9     | 0.065367   | 0.257863   | 0.934371 | 0.065629   | 01:26  |

- **Model Training**: Over 10 epochs, the model learns and adapts to the pet breed classification task, progressively improving its accuracy.

- **Performance Metrics**: Comprehensive metrics, including training and validation loss, accuracy, and error rate, highlight the model's progress and proficiency in classifying pet breeds.

- **Training Time**: Each epoch consistently takes approximately 1 minute and 27 seconds.

- **Model Preservation**: The trained model is saved as 'model1_freezed,' preserving both its architecture and learned weights for further evaluation and deployment.

### 6. Model Interpretation
---
- Top 10 Metrics from Classification Report:

| Breed Category            | Precision | Recall | F1-Score |
|---------------------------|-----------|--------|----------|
| Abyssinian                | 0.86      | 0.93   | 0.89     |
| Bengal                    | 0.92      | 0.73   | 0.81     |
| Siamese                   | 0.90      | 1.00   | 0.95     |
| Birman                    | 0.90      | 0.92   | 0.91     |
| Bombay                    | 0.98      | 0.98   | 0.98     |
| British_Shorthair         | 0.94      | 0.80   | 0.86     |
| Ragdoll                   | 0.81      | 0.91   | 0.86     |
| Maine_Coon                | 0.90      | 0.90   | 0.90     |
| Persian                   | 0.97      | 0.85   | 0.91     |
| Russian_Blue              | 0.79      | 0.94   | 0.86     |

- **Most Confused Categories**:

| Category Pair                        | Confusion Count |
|-------------------------------------|-----------------|
| British_Shorthair vs. Russian_Blue  | 5               |
| Beagle vs. Basset_Hound             | 5               |
| Bengal vs. Abyssinian               | 4               |
| Persian vs. Ragdoll                 | 4               |
| Ragdoll vs. Birman                  | 4               |
| Chihuahua vs. Miniature_Pinscher    | 4               |
| Bengal vs. Maine_Coon               | 3               |
| Birman vs. Siamese                  | 3               |
| Maine_Coon vs. Ragdoll              | 3               |
| American_Pit_Bull_Terrier vs. Miniature_Pinscher | 3 |

- The classification report provides precision, recall, and F1-score for each pet breed category, offering a detailed view of the model's performance.
- The most confused categories shed light on breed pairs that the model frequently struggles to distinguish.

### 7. Unfreezing Model Layers, Fine-Tuning & Learning Rate
---
**Previous Model Training (Frozen Layers)**

| Epoch | Train Loss | Valid Loss | Accuracy | Error Rate | Time   |
|-------|------------|------------|----------|------------|--------|
| 0     | 0.666741   | 0.321287   | 0.895129 | 0.104871   | 01:27  |
| 1     | 0.496385   | 0.430292   | 0.875507 | 0.124493   | 01:26  |
| 2     | 0.483526   | 0.561120   | 0.868065 | 0.131935   | 01:27  |
| 3     | 0.375414   | 0.347090   | 0.908660 | 0.091340   | 01:28  |
| 4     | 0.289794   | 0.372382   | 0.899188 | 0.100812   | 01:27  |
| 5     | 0.215737   | 0.319737   | 0.920839 | 0.079161   | 01:25  |
| 6     | 0.156200   | 0.319586   | 0.924899 | 0.075101   | 01:27  |
| 7     | 0.110415   | 0.235808   | 0.936401 | 0.063599   | 01:27  |
| 8     | 0.078930   | 0.260270   | 0.929635 | 0.070365   | 01:27  |
| 9     | 0.065367   | 0.257863   | 0.934371 | 0.065629   | 01:26  |

**Fine-Tuned Model (Unfreezed Layers)**

| Epoch | Train Loss | Valid Loss | Accuracy | Error Rate | Time   |
|-------|------------|------------|----------|------------|--------|
| 0     | 0.057250   | 0.259445   | 0.929635 | 0.070365   | 01:26  |
| 1     | 0.052452   | 0.261673   | 0.934371 | 0.065629   | 01:26  |
| 2     | 0.043833   | 0.252830   | 0.935047 | 0.064953   | 01:25  |
| 3     | 0.050001   | 0.279817   | 0.933694 | 0.066306   | 01:26  |
| 4     | 0.044332   | 0.257765   | 0.932341 | 0.067659   | 01:26  |
| 5     | 0.043266   | 0.263906   | 0.937077 | 0.062923   | 01:27  |
| 6     | 0.043428   | 0.253806   | 0.934371 | 0.065629   | 01:28  |
| 7     | 0.032019   | 0.250571   | 0.937754 | 0.062246   | 01:29  |
| 8     | 0.035151   | 0.254164   | 0.936401 | 0.063599   | 01:14  |
| 9     | 0.036221   | 0.245009   | 0.935047 | 0.064953   | 01:03  |


**Comparison**:

1. **Training Loss**: In the previous model with frozen layers, the training loss started at 0.667 and gradually decreased to 0.065 in 10 epochs. After unfreezing and fine-tuning, the training loss starts at 0.057 and ends at 0.036. The fine-tuned model exhibits lower training loss, indicating better convergence and learning.

2. **Validation Loss**: Similar to training loss, validation loss also decreased from 0.321 to 0.258 in the previous model. In the fine-tuned model, it decreases from 0.259 to 0.245. The fine-tuned model maintains a lower validation loss, showing improved generalization.

3. **Accuracy**: The fine-tuned model, however, starts with an accuracy of 92.9% and reaches 93.5%. While the difference is relatively small, it indicates a slight improvement in correctly classifying images.

4. **Error Rate**: The error rate, inversely related to accuracy, improved from 10.5% to 6.6% in the previous model. In the fine-tuned model, it decreased from 7.0% to 6.5%. Although the change is modest, it demonstrates the fine-tuned model's enhanced precision in classifying pet breeds.

5. **Training Time**: The training time for each epoch remains consistent in both models, approximately 1 minute and 26 seconds. Fine-tuning did not significantly impact the computational efficiency of the training process.

## Conclusion
---
In this Fastai computer vision project, we developed a highly sophisticated Cat & Dog breed detector using the state-of-the-art Fastai framework. Our goal was to accurately classify pet images into 37 distinct categories, representing various breeds, using the challenging Oxford-IIIT Pet Dataset.

The project was structured into key phases, each contributing to the success of our computer vision model:

1. **Load Data**: We acquired the Oxford-IIIT Pet Dataset and prepared it for model training.

2. **Data Preparation**: We performed data normalization, image augmentation, and defined item and batch-level transformations to enhance dataset diversity.

3. **Create DataLoader**: Data loaders were created to efficiently load and batch the data for training and validation.

4. **Transfer Learning with Pretrained Model**: Leveraging the power of transfer learning, we started with a pretrained model as the foundation of our Cat & Dog breed detector. Specifically, we used a ResNet-50 architecture pretrained on the ImageNet dataset. This allowed us to harness the knowledge of the pretrained model to excel in our specific task.

5. **Train & Save Model**: Our model underwent training for 10 epochs, achieving impressive accuracy and precision. The trained model was saved for future use.

6. **Model Interpretation**: We analyzed the model's performance, examining classification metrics and identifying the most confused categories.

7. **Unfreeze Model Layers, Fine-Tune & Learning Rate**: We fine-tuned the model by unfreezing layers, resulting in improved training and validation loss, accuracy, and error rate.

This project demonstrates the power of Fastai in developing state-of-the-art computer vision models. Through meticulous data preparation, model definition, and fine-tuning, we achieved remarkable accuracy in the challenging task of pet breed classification. The lessons learned and insights gained from this project can be applied to a wide range of computer vision applications, paving the way for further advancements in the field.
