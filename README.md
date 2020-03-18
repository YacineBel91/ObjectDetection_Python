# ObjectDetection_Python
# Install Python 3.7
# Install tensorflow (if this gives problems may be bc of the permissions, so change them or install everything in a virtual environment)
pip install tensorflow
# Install OpenCV
pip install opencv-python
# Install Keras
pip install keras
# Install ImageAI
pip install imageAI
# Download TinyYOLOv3 or YOLOv3, or any other model file that contains the classification model that will be used for object detection
# Step 1_ Create root folder called: ObjectDetection
# Step 1.2_ In ObjectDetection, create three folders: input (will contain image to analize), output (image analyzed), model (TinyYOLOv3, in this case)
# Step 2_Create a new file 'detector.py' to writing the code
# Step 3_Import ObjectDetection class from the ImageAI library
from imageai.Detection import ObjectDetection
# Step 4_Create an instance of the class ObjectDetection
detector = ObjectDetection()
# Step 5_Specify the path from our input image (name.format), output image, and model
model_path = "./model/yolo-tiny.h5"
input_path = "./input/test45.jpg"
output_path = "./output/newimage.jpg"
# Step 6_ Load our model
detector.setModelTypeAsTinyYOLOv3()
# Step 7_Call the function 'setModelPath()'. This function accepts a string which contains the path to the pre-trained model
detector.setModelPath(model_path)
# Step 8_Call the function 'loadModel()' from the detector instance. And load the model from the path specified above using the 'setModelPath()' class method
detector.loadModel()
# Step 9_Call the 'detectObjectsFromImage' function using the detector object that we created in the previous section. This function requires two arguments: 'input_image' and 'output_image_path'. input_image is the path where the image we are detecting is located, while the output_image_path parameter is the path to store the image with detected objects. This function returns a dictionary which contains the names and percentage probabilities of all the objects detected in the image.
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)
# Step 10_Output
for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])
    
# Output of the example
car  :  54.72719073295593
car  :  58.94589424133301
car  :  62.59384751319885
car  :  74.07448291778564
car  :  91.10507369041443
car  :  97.26507663726807
car  :  97.55765795707703
person  :  53.6459743976593
person  :  56.59831762313843
person  :  72.28181958198547

# Complete Code for Object Detection

from imageai.Detection import ObjectDetection

detector = ObjectDetection()

model_path = "./model/yolo-tiny.h5"
input_path = "./input/test45.jpg"
output_path = "./output/newimage.jpg"

detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])
    
# References
https://imageai.readthedocs.io/en/latest/detection/index.html
https://stackabuse.com/object-detection-with-imageai-in-python/
https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606
https://www.spiedigitallibrary.org/journals/journal-of-applied-remote-sensing/volume-13/issue-04/044509/Underwater-and-airborne-monitoring-of-marine-ecosystems-and-debris/10.1117/1.JRS.13.044509.full?SSO=1
https://arxiv.org/abs/2001.00361

    
    
    
    
    
    
    
    
    
    
    
