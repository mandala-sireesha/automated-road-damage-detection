from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
from django.core.files.storage import FileSystemStorage

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})

def training(request):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    # Set up directories
    train_dir = r'C:\Users\dell in 3530\OneDrive\Desktop\Automated_Road_Damage_Detection\media\train'
    validation_dir = r'C:\Users\dell in 3530\OneDrive\Desktop\Automated_Road_Damage_Detection\media\validation'

    # ImageDataGenerator for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # ImageDataGenerator for validation (only rescaling)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    # Build the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # 4 classes
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
    val_accuracy = f'Validation accuracy: {accuracy * 100:.2f}%'

    # Save the model
    model.save('road_classification_model.h5')

    return render(request, 'users/training.html',{'loss':loss,'accuracy':accuracy,'val_accuracy':val_accuracy})

# def PredictRoadDamage(request):
#     import os
#     from django.conf import settings
#     from django.core.files.storage import FileSystemStorage
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import matplotlib.image as mpimg
#     from tensorflow.keras.models import load_model
#     from tensorflow.keras.preprocessing import image
#     import cv2
#     from PIL import Image
#     from django.shortcuts import render

#     if request.method == 'POST':
#         image_file = request.FILES['file']
#         fs = FileSystemStorage(location="media/sih_road_dataset/test_data")
#         filename = fs.save(image_file.name, image_file)
#         uploaded_file_url = "/media/sih_road_dataset/test_data/" + filename
#         path = os.path.join(settings.MEDIA_ROOT, 'sih_road_dataset/test_data', filename)

#         # Load the model
#         model_path = os.path.join(settings.MEDIA_ROOT, 'Road_Damage_model.h5')
#         model = load_model(model_path)

#         # Load and preprocess the image
#         im = cv2.imread(path)  # Corrected to use 'path'
#         img = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))  # Corrected color conversion
#         img = img.resize((224, 224))  # Adjust target_size to match your model's input size
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)  # Create a batch

#         # Preprocess the image (normalize)
#         img_array /= 255.0  # Example normalization; adjust based on your model's requirements

#         # Debugging: Print preprocessed image data
#         print("Preprocessed image data shape:", img_array.shape)
#         print("Preprocessed image data values (sample):", img_array[0][0][0])

#         # Make a prediction
#         predictions = model.predict(img_array)
#         print("Raw model predictions:", predictions)  # Debugging: Print raw predictions
#         predicted_class = np.argmax(predictions, axis=1)[0]  # Get the predicted class index
#         print('Predicted class:', predicted_class)

#         # Map predicted class to a profile name
#         profile_name = "Unknown"  # Default value
#         if predicted_class == 0:
#             profile_name = "very poor"
#         elif predicted_class == 1:
#             profile_name = "Poor"
#         elif predicted_class == 2:
#             profile_name = "Satisfactory"
#         elif predicted_class == 3:
#             profile_name = "good"
#         result = f"{path} is {profile_name}"

#         # Display the image
#         img = mpimg.imread(path)
#         plt.imshow(img)
#         plt.title(f'Prediction: {profile_name}')
#         plt.show()

#         return render(request, "users/UploadForm.html", {'path': uploaded_file_url, 'result': result})
#     else:
#         return render(request, "users/UploadForm.html", {})

def PredictRoadDamage(request):
    import os
    from django.conf import settings
    from django.core.files.storage import FileSystemStorage
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import cv2
    from PIL import Image
    from django.shortcuts import render

    if request.method == 'POST':
        image_file = request.FILES['file']
        fs = FileSystemStorage(location="media/prediction_data")
        filename = fs.save(image_file.name, image_file)
        uploaded_file_url = "/media/prediction_data/" + filename
        path = os.path.join(settings.MEDIA_ROOT, 'prediction_data', filename)

        # Load the model
        model_path = os.path.join(settings.MEDIA_ROOT, 'road_classification_model.h5')
        model = load_model(r'C:\Users\dell in 3530\OneDrive\Desktop\Automated_Road_Damage_Detection\road_classification_model.h5')

        # Load and preprocess the image
        im = cv2.imread(path)
        if im is None:
            print(f"Error loading image: {path}")
            return render(request, "users/UploadForm.html", {'result': 'Error loading image'})

        img = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        img = img.resize((150, 150))  # Resize to match your model's expected input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Create a batch

        # Preprocess the image (normalize)
        img_array /= 255.0  # Example normalization; adjust based on your model's requirements

        # Debugging: Print preprocessed image data
        print("Preprocessed image data shape:", img_array.shape)
        print("Preprocessed image data values (sample):", img_array[0][0][0])

        # Make a prediction
        predictions = model.predict(img_array)
        print("Raw model predictions:", predictions)  # Debugging: Print raw predictions
        predicted_class = np.argmax(predictions, axis=1)[0]  # Get the predicted class index
        print('Predicted class:', predicted_class)

        # Map predicted class to a profile name
        profile_name = "Unknown"  # Default value
        class_mappings = {0: "good", 1: "poor", 2: "satisfactory", 3: "very_poor"}
        
        if predicted_class in class_mappings:
            profile_name = class_mappings[predicted_class]
        else:
            print(f"Unexpected class index: {predicted_class}")

        result = f"{path} is {profile_name}"

        # Display the image
        img = mpimg.imread(path)
        plt.imshow(img)
        plt.title(f'Prediction: {profile_name}')
        plt.show()

        return render(request, "users/UploadForm.html", {'path': uploaded_file_url, 'result': result})
    else:
        return render(request, "users/UploadForm.html", {})




