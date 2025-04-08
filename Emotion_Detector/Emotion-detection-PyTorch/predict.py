import cv2
import torch
from FaceCNNModel import FaceCNN
import torchvision.transforms as transforms
import torch.nn.functional as F
nb_classes = 7
model = FaceCNN(nb_classes)

state_dict  = torch.load('pytorch_weights/best weights.pth',map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def detect_emotions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        # No faces detected, return the frame without any predictions
        return frame

    for (x, y, w, h) in faces:
        # Obtaining the face coordinates
        image = gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Resizing the face image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        try:
            # Transform the image and add a batch dimension
            image = transform(image).unsqueeze(0)

            labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

            # Setting the model to evaluation mode
            model.eval()

            # Disabling gradient calculation
            with torch.no_grad():
                logits = model(image)
                probs = F.softmax(logits, dim=1)
                prediction_label = labels[probs.argmax().item()]

                
                print(f"Predicted Label: {prediction_label}")

            # Display the predicted emotion on the frame
            cv2.putText(frame, '%s' % (prediction_label), (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        except Exception as e:
            print(f"Error processing face: {e}")

    return frame