# Face Gaze
Inferring on screen gaze location using an image of the Viewer's face.
## Model Architecture
It's a multimodal CNN, that takes, as input, 
* Image of Face
* Image of left eye
* Image of right eye
* Location of face in image

And gives as output, the x and y coordinates of the user's focus point on the screen.

It looks like this
![Model Architecture](https://github.com/aayush-fadia/face-gaze/raw/master/media/multi_input_and_output_model.png)
## Results
These are from the test set, this person was not in the training set.
![Results GIF](https://github.com/aayush-fadia/face-gaze/raw/master/media/ModelV2.gif)
