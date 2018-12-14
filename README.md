# EmojiMe
EmojiMe is a program that allows users to insert an emoji at the location of their cursor by mimicking the emoji with their face.

## Requirements
  - Flask 1.0.2
  - dlib 19.16.0
  - matplotlib 3.0.0
  - opencv_python 3.4.3.18
  - numpy 1.15.2
  - scikit_learn 0.20.1

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
 
## Files
  - `/app.py` is the main file of the program.
  
 #TODO finish up files
