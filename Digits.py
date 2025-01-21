# Import necessary libraries
import sys  # For command-line arguments
import os.path  # For file path operations
import cv2  # OpenCV for image processing
import numpy as np  # For numerical operations
from skimage.feature import hog  # For HOG (Histogram of Oriented Gradients) feature extraction
from sklearn import datasets  # Import datasets module from sklearn
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree classifier
from sklearn.neighbors import KNeighborsClassifier  # Import k-NN classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # Import LDA classifier
from sklearn.naive_bayes import GaussianNB  # Import Naive Bayes classifier
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest classifier
from sklearn.svm import SVC  # Import Support Vector Machine classifier
from joblib import dump, load  # Import joblib for saving/loading models

# Create the classifier object based on index
def createClassifierObject(clfIndex):
    if clfIndex == 0:
        return DecisionTreeClassifier()
    elif clfIndex == 1:
        return KNeighborsClassifier()
    elif clfIndex == 2:
        return LinearDiscriminantAnalysis()
    elif clfIndex == 3:
        return GaussianNB()
    elif clfIndex == 4:
        return RandomForestClassifier()
    elif clfIndex == 5:
        return SVC()
    else:
        print("Unknown Classifier Index!")
        exit(-1)

# Get the filename of the trained classifier data
def getTrainedDataFile(clfIndex):
    filenames = ["dtc.clf", "knn.clf", "lda.clf", "nb.clf", "rfc.clf", "svm.clf"]
    print(f"Looking for trained data of classifier index {clfIndex}...")
    return filenames[clfIndex] if 0 <= clfIndex < len(filenames) else exit(-1)

# Load classifier, train if necessary, and save it
def loadClassifier(clfIndex):
    filename = getTrainedDataFile(clfIndex)
    if os.path.isfile(filename):
        print("Trained data is available! Loading the classifier...")
        return load(filename)
    else:
        print("Trained data is not available! Loading the MNIST dataset...")
        dataset = datasets.fetch_openml("mnist_784", version=1)
        features = np.array(dataset.data, "int16")
        labels = np.array(dataset.target, "int")

        # Extract HOG features
        hogFD = [hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1)) for feature in features]
        hogFeatures = np.array(hogFD, "float64")

        # Train the classifier
        clf = createClassifierObject(clfIndex)
        print("Training the classifier...")
        clf.fit(hogFeatures, labels)
        
        # Save trained model
        print("Saving the trained data...")
        dump(clf, filename, compress=3)
        return clf

# Process the input image and perform digit recognition
def performRecognition(clf, img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    imgGray = cv2.GaussianBlur(imgGray, (5, 5), 0)  # Apply Gaussian blur
    _, imgThresh = cv2.threshold(imgGray, 90, 255, cv2.THRESH_BINARY_INV)  # Thresholding
    ctrs, _ = cv2.findContours(imgThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours

    for rect in [cv2.boundingRect(ctr) for ctr in ctrs]:
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
        leng = int(h * 1.6)
        roi = imgThresh[max(0, y + h // 2 - leng // 2):min(imgThresh.shape[0], y + h // 2 + leng // 2), max(0, x + w // 2 - leng // 2):min(imgThresh.shape[1], x + w // 2 + leng // 2)]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)  # Resize to 28x28
        roi = cv2.dilate(roi, (3, 3))
        roi_hogFD = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))  # Compute HOG
        nbr = clf.predict(np.array([roi_hogFD], 'float64'))
        cv2.putText(img, str(int(nbr[0])), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)

    cv2.imshow("RESULT", img)
    print("Press any key to exit!")
    cv2.waitKey()
    cv2.destroyAllWindows()

# Main function
def main():
    if len(sys.argv) < 3:
        print("Invalid arguments!\nUsage: python %s <Classifier Index> <Test Image>" % sys.argv[0])
        print("Classifier Index:")
        print("0 - Decision Tree\n1 - k-NN\n2 - LDA\n3 - Naive Bayes\n4 - Random Forest\n5 - SVM")
        exit(-1)

    clfIndex = int(sys.argv[1])
    clf = loadClassifier(clfIndex)  # Load classifier

    if os.path.isfile(sys.argv[2]):
        print("Reading the test image...")
        img = cv2.imread(sys.argv[2])
    else:
        print(f"{sys.argv[2]} does not exist!")
        exit(-1)
    
    print("Performing digit recognition...")
    performRecognition(clf, img)

if __name__ == '__main__':
    main()
