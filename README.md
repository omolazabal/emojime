# EmojiMe
EmojiMe is a program that allows users to insert an emoji at the location of their cursor by mimicking the emoji with their face.

## Requirements
  - Flask 1.0.2
  - dlib 19.16.0
  - matplotlib 3.0.0
  - opencv_python 3.4.3.18
  - numpy 1.15.2
  - scikit_learn 0.19.1

## Installation  
1. Download the source code of the project by running `git clone https://github.com/omolazabal/emojime.git` in your shell.
2. [Install Python 3](https://www.python.org/downloads/)
3. Navigate to the root of the EmojiMe source code and run `pip install -r requirements.txt` in your shell to install most of the dependicies.
4. Install Dlib
    - [Unix Based OS](https://www.learnopencv.com/install-dlib-on-macos/)
    - [Windows](https://www.learnopencv.com/install-dlib-on-windows/)
    
## Run
To run the application in standard mode navigate to the root of the EmojiMe source code and run the command `python3 app.py` in your shell.

## Modes
EmojiMe contains three modes. You can run each mode by specifying certain flags upon running the program:
 - No flag enters standard mode
 - `--debug` enters debug mode.
 - `--data` enters data generation mode.

**Standard Mode.** This mode is used to run the standard application.

To run the application in standard mode navigate to the root of the EmojiMe source code and run the command `python3 app.py` in your shell. You will be greeted with a input text box and a button that allows you to insert an emoji at the location of your cursor. 

**Debug Mode.** This mode is used to view how the image processing works behind the scenes.  

To run the application in debug mode navigate to the root of the EmojiMe source code and run the command `python3 app.py --debug` in your shell. You will be greeted with an web cam feed along with the features that are available in standard mode.

**Data Generation Mode.** Data generation mode is used to generate training data.  

To run the application in data generation mode navigate to the root of the EmojiMe source code and run the command `python3 app.py --data --type <parameter 1> --start <parameter 2>` in your shell. You will be greeted with a webcam feed and a button to capture the face in the feed. The data will be saved in the `/data` directory.
 - `<parameter 1>` should specify the emotion you will be providing training data for. Valid parameters for parameter 1 are `angry` `sad` `happy` `fear` `neutral`.
  - `<parameter 2>` should specify the starting index of your image. The index are used to name the files. For instance if you specify `3` for parameter 2, your images will be saved as `3.png`, `4.png`, ... , `n.png` where n represents the index of the last image you took. Valid parameters for parameter 2 are any positive integer value.
 
## Files and directories
  - `/app.py` is the main file of the program.
    - Launches the flask server and acts as the middle-man between the webpage and the resources needed to run the program. Utilizes the `emojime` module to initialize a face detector which is the basis of the program. 
  - `/emojime` is the module utilized to perform all back-end actions.
    - `emojime/face_detector.py` contains the class definiton for `FaceDetector`. A `FaceDetector` object is utilized to perform image processing on the user's face. The object is able to extract the user's face, apply landmarks, make predictions, and save training data.
  - `/templates` is the directory that contains the html pages for the different modes of the application. [UIKit](https://getuikit.com/) is utilized to design the different webpages.
  - `/models` contains the trained models that are utilized for this project
    - `/models/emotion_scaler` contains the standard deviation and mean of the traning data. It is utilized to standardize the data predictions will be performed on.
    - `/models/svm_emotion_classifier` contains the model that is used to predict which emotion a given image has. 
    - `/models/shape_predictor_68_face_landmarks.dat` contains the model that is utilized to extract 68 facial landmarks of a given image.
  - `/data` is a directory that contains the training data for the `/models/svm_emoition_classifier` model
  - `/notebooks` contains the Jupyter Notebooks that were utilized to create the `models/svm_emotion_classifier` model.
    - `/notebooks/create_data.ipynb` is utilized to extract the features from the images the model will be training off of. It extracts the landmarks, calulates the euclidean distance between each landmark, and formats the training data into two Numpy arrays `X` and `y`. `X` consists of training samples and `y` contains the corresponding labels.
    - `/notebooks/train_model.ipynb` contains the code that is utilized to train the `/models/svm_emotion_classifier` model. It creates a train-test split, evaluates the performance of the model using the testing data, and the trains the a model using the entire dataset.
  
