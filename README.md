\# Human Activity Recognition Using Deep Learning



\## Project Overview



This project focuses on Human Activity Recognition (HAR) using deep learning techniques applied to wearable sensor data. The system analyzes multivariate time-series signals collected from accelerometer and gyroscope sensors to classify different human activities such as walking, sitting, standing, laying, walking upstairs, and walking downstairs.



The project explores the use of recurrent neural networks (LSTM) and hybrid CNN-LSTM architectures for learning temporal motion patterns from raw sensor signals. In addition to model training and evaluation, the project includes confusion matrix analysis, classification reports, model comparison, and an interactive Streamlit application for real-time activity prediction visualization.



The broader motivation of this work is related to intelligent healthcare systems and wearable AI technologies, where human activity recognition can support applications such as elderly monitoring, rehabilitation, cognitive-motor analysis, and future sensor-based health platforms such as HEG (Health Entertainment Gadget).



\## Dataset



The project uses the UCI Human Activity Recognition (HAR) Dataset, which contains wearable sensor data collected from smartphones mounted on participants' waists. The dataset includes accelerometer and gyroscope measurements recorded during six different human activities:



\- WALKING

\- WALKING\_UPSTAIRS

\- WALKING\_DOWNSTAIRS

\- SITTING

\- STANDING

\- LAYING



The sensor signals were sampled at approximately 50 Hz and segmented into fixed-size windows of 128 timesteps (approximately 2.56 seconds per sample).



The final input tensor shape used in this project was:



```text

(samples, timesteps, channels)

(7352, 128, 9)



\## Problem Statement



Human Activity Recognition (HAR) is an important area in wearable computing and intelligent healthcare systems. The goal of HAR is to automatically identify human activities using sensor data collected from wearable devices.



Accurate activity recognition can support many real-world applications, including:



\- elderly monitoring,

\- rehabilitation systems,

\- fitness tracking,

\- fall detection,

\- smart healthcare platforms,

\- and cognitive-motor behavior analysis.



One of the main challenges in HAR is distinguishing between activities with similar motion characteristics. Dynamic activities such as walking, walking upstairs, and walking downstairs may produce similar temporal patterns, while static activities such as sitting and standing often generate comparable inertial sensor signals.



This project investigates how deep learning architectures such as LSTM and CNN-LSTM can learn temporal motion representations from raw multivariate sensor data and how different architectures affect classification behavior and activity confusion patterns.



\## Methodology



The project was implemented using a deep learning workflow for multivariate time-series classification.



\### Data Preparation



The raw sensor signals from the UCI HAR dataset were loaded and merged into a three-dimensional tensor structure:



```text

(samples, timesteps, channels)

(7352, 128, 9)



The nine sensor channels included:



Body acceleration (X, Y, Z)

Gyroscope signals (X, Y, Z)

Total acceleration signals (X, Y, Z)



\## Exploratory Data Analysis

Exploratory Data Analysis (EDA)



Several activity samples and raw sensor signals were visualized to better understand temporal motion patterns across different activities. Dynamic activities showed periodic motion patterns, while static activities demonstrated lower signal variation.





\## Model Architectures

Deep Learning Models



Two deep learning architectures were implemented and compared:



Baseline LSTM Model

CNN-LSTM Hybrid Model



The LSTM model focused on learning temporal dependencies directly from the raw sensor sequences.



The CNN-LSTM model combined convolutional feature extraction with temporal sequence learning to investigate whether local pattern extraction could improve classification performance.



Model Evaluation



The models were evaluated using:



Validation accuracy

Test accuracy

Test loss

Confusion matrices

Classification reports

Precision, recall, and F1-score analysis



The evaluation process focused not only on overall accuracy but also on understanding activity confusion behavior and architectural differences between the models.





\## Results



The baseline LSTM model achieved approximately:



\- 89.9% validation accuracy

\- 88.97% test accuracy



The CNN-LSTM hybrid model achieved:



\- 91.3% validation accuracy

\- 88.46% test accuracy



Although the CNN-LSTM model reached a slightly higher validation accuracy, the baseline LSTM model produced more reliable classification behavior on the unseen test data.



The confusion matrix and classification reports revealed several important observations:



\- Dynamic activities such as walking-related movements were generally classified accurately due to their stronger temporal motion patterns.

\- Walking, walking upstairs, and walking downstairs remained partially difficult to separate because of similarities in their movement sequences.

\- Sitting and standing, which are both relatively static activities, showed noticeable classification confusion due to similar inertial sensor characteristics.

\- The laying activity was recognized very accurately because of its distinct body orientation and sensor pattern.



An important finding was that the CNN-LSTM architecture introduced additional confusion between some static and dynamic activities compared to the baseline LSTM model. This suggests that convolutional pooling may reduce certain temporal distinctions necessary for precise activity separation.



These results demonstrate that higher validation accuracy alone does not necessarily indicate better real-world classification performance or interpretability.



\## Model Comparison



Two deep learning architectures were implemented and compared in this project:



| Model | Validation Accuracy | Test Accuracy | Key Observation |

|---|---|---|---|

| Baseline LSTM | \~89.9% | \~88.97% | Better temporal separation and more consistent classification behavior |

| CNN-LSTM | \~91.3% | \~88.46% | Higher validation accuracy but increased confusion between certain activities |



The baseline LSTM model demonstrated stronger temporal activity separation, especially between dynamic and static activities.



Although the CNN-LSTM architecture achieved slightly higher validation accuracy, it introduced additional confusion between activities with different behavioral characteristics. In particular, some static activities such as sitting and standing were occasionally confused with movement-related activities more frequently than in the baseline LSTM model.



This comparison highlights the importance of evaluating deep learning architectures not only by overall accuracy, but also by confusion behavior, interpretability, and activity-level classification consistency.



\## Streamlit Application



An interactive Streamlit application was developed as part of the project to visualize model predictions and sensor signals in real time.



The application allows users to:



\- Select test samples interactively

\- Visualize raw sensor signals

\- Display predicted and true activity labels

\- Show prediction confidence scores

\- Explore activity recognition behavior across different samples



The Streamlit application provides a practical demonstration of how deep learning models can be integrated into interactive AI-based healthcare and wearable sensing systems.



\## Conclusion



This project demonstrated the effectiveness of deep learning techniques for Human Activity Recognition using wearable sensor data.



Two deep learning architectures, a baseline LSTM model and a CNN-LSTM hybrid model, were implemented and compared using raw multivariate time-series signals from the UCI HAR dataset.



The results showed that both models were capable of learning meaningful temporal motion patterns and achieved strong overall classification performance. The baseline LSTM model produced more consistent activity separation on unseen test data, while the CNN-LSTM model achieved slightly higher validation accuracy but introduced additional confusion between certain activities.



The analysis also highlighted the importance of evaluating model behavior beyond overall accuracy. Confusion matrices and classification reports revealed that activities with similar motion or posture characteristics remain challenging to separate, particularly sitting and standing activities.



In addition to the deep learning models, an interactive Streamlit application was developed to visualize sensor signals and perform real-time activity prediction demonstrations.



This project provides a strong foundation for future wearable AI and intelligent healthcare systems, including applications related to rehabilitation, elderly monitoring, activity analysis, and future sensor-based platforms such as HEG (Health Entertainment Gadget).



\## How to Run the Project



\### 1. Clone or Download the Project



Place the project files in a local directory with the following structure:



```text

project/

├── streamlit\_app/

├── data/

├── models/

├── notebooks/

├── outputs/

└── README.md



2\. Install Required Libraries



Install the required Python packages:

pip install numpy pandas matplotlib tensorflow scikit-learn streamlit



3\. Prepare the Dataset



Download the UCI HAR Dataset and place it inside the data/ directory:

data/UCI HAR Dataset/



4\. Run the Jupyter Notebook



Open the notebook and execute all cells to:



load the dataset,

preprocess the signals,

train the models,

evaluate the results,

and save the trained models.





5\. Run the Streamlit Application



Open a terminal in the project directory and run:

streamlit run streamlit\_app/app.py



6\. Use the Application



The Streamlit application allows users to:



select activity samples,

visualize sensor signals,

compare predicted and true labels,

and explore deep learning activity recognition behavior interactively.



#   D e e p - L e a r n i n g  
 