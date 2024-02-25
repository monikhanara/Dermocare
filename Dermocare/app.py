import json
import io
import os
import base64
import numpy as np
import streamlit as st
from streamlit import session_state
from PIL import Image
import matplotlib.pyplot as plt
import shap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing import image as img_preprocessing
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array


session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0


def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")

        if st.form_submit_button("Signup"):
            if password == confirm_password:
                user = create_account(name, email, age, sex, password, json_file_path)
                session_state["logged_in"] = True
                session_state["user_info"] = user
            else:
                st.error("Passwords do not match. Please try again.")


def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
                render_dashboard(user)
                return user

        st.error("Invalid credentials. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None


def predict_skin(image_path, model):
    img = Image.open(image_path).resize((32, 32))
    img_array = img_preprocessing.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions using the model
    predictions = model.predict(img_array)

    # Get the predicted label
    predicted_label_index = np.argmax(predictions)

    # Define class labels
    class_labels = {
        0: "Actinic keratoses",
        1: "Basal cell carcinoma",
        2: "Benign keratosis-like lesions",
        3: "Dermatofibroma",
        4: "Melanoma",
        5: "Melanocytic nevi",
        6: "Vascular lesions"
    }

    predicted_label = class_labels[predicted_label_index]

    return predicted_label

def load_background_batch():
    dir_path = './data'
    batch_data = []
    image_files = os.listdir(dir_path)
    # print(image_files)

    background_data = []

    for img_file in image_files:
        img_path = os.path.join(dir_path, img_file)
        # print(img_path)
        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        background_data.append(img_array)

    background_batch = np.vstack(background_data)

    batch_data.append(background_batch)

    return batch_data


def shap_explanation(model, img_array, background):
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(img_array)
    return shap_values


def show_shap(shap_values, img_array, predicted_class_idx):
    
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)

    # Get the SHAP values for the predicted class
    shap_values_for_predicted_class = shap_values[predicted_class_idx]

    # Plotting
    plt.figure()
    shap.image_plot(shap_values_for_predicted_class, img_array)
    plt.show()
    


    st.write('Inference for the Prediction: Plot of SHAP values')
    st.pyplot(plt.gcf())


def get_predictions(uploaded,model):
    if uploaded is not None:
        global img
        img = Image.open(uploaded).resize((32,32))
        img_array = img_to_array(img)/255.0
        img_array = np.expand_dims(img_array,axis=0)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        print(predicted_class)
        confidence = prediction[0, predicted_class]
        return predicted_class, confidence, img_array

def generate_medical_report(predicted_label):

    skin_disease_info = {
        "Actinic keratoses": {
            "report": "It appears the patient may have actinic keratoses, which are precancerous skin lesions caused by sun damage. Early treatment is crucial to prevent progression to skin cancer.",
            "preventative_measures": [
                "Avoid prolonged sun exposure, especially during peak hours",
                "Use sunscreen with a high SPF regularly, and reapply as needed",
                "Wear protective clothing, hats, and sunglasses when outdoors",
            ],
            "precautionary_measures": [
                "Schedule regular skin checks with a dermatologist",
                "Seek medical attention if any lesions show signs of change or growth",
            ],
        },
        "Basal cell carcinoma": {
            "report": "The patient may be showing signs of basal cell carcinoma, the most common form of skin cancer. Early detection and treatment are crucial for a good prognosis.",
            "preventative_measures": [
                "Protect skin from sun exposure with clothing and sunscreen",
                "Avoid indoor tanning beds and booths",
                "Perform regular self-examinations of the skin to detect any changes",
            ],
            "precautionary_measures": [
                "Consult with a dermatologist for further evaluation and treatment options",
                "Follow recommended follow-up appointments to monitor for recurrence or new lesions",
            ],
        },
        "Benign keratosis-like lesions": {
            "report": "The patient may have benign keratosis-like lesions, which are non-cancerous growths often caused by sun exposure. While typically harmless, monitoring for changes is recommended.",
            "preventative_measures": [
                "Protect skin from UV radiation with sunscreen and protective clothing",
                "Keep skin moisturized to prevent dryness and irritation",
                "Avoid picking or scratching at lesions to prevent infection",
            ],
            "precautionary_measures": [
                "Consult with a dermatologist for a proper diagnosis and management plan",
                "Keep track of any changes in size, color, or texture of the lesions for monitoring purposes",
            ],
        },
        "Dermatofibroma": {
            "report": "It seems the patient may have dermatofibroma, a benign skin growth commonly found on the legs. While typically harmless, it's important to monitor for changes.",
            "preventative_measures": [
                "Avoid unnecessary trauma or injury to the skin",
                "Keep skin moisturized to prevent irritation and itching",
                "Regularly inspect the skin for any changes in the appearance or texture of the growths",
            ],
            "precautionary_measures": [
                "Consult with a dermatologist for confirmation and advice on management",
                "Monitor for any changes in size, color, or texture of the dermatofibromas",
            ],
        },
        "Melanoma": {
            "report": "There are indications of melanoma, the most serious form of skin cancer. Immediate medical attention and further evaluation are necessary for proper diagnosis and treatment.",
            "preventative_measures": [
                "Protect skin from UV radiation by seeking shade and wearing sunscreen",
                "Perform regular self-examinations of the skin to detect any new or changing moles",
                "Avoid indoor tanning beds and booths",
            ],
            "precautionary_measures": [
                "Seek urgent evaluation by a dermatologist or oncologist for biopsy and treatment planning",
                "Follow recommended surveillance protocols for monitoring and follow-up appointments",
            ],
        },
        "Melanocytic nevi": {
            "report": "It appears the patient may have melanocytic nevi, commonly known as moles. While usually benign, changes in size, shape, or color should be monitored closely.",
            "preventative_measures": [
                "Be vigilant about changes in existing moles or the appearance of new ones",
                "Protect skin from UV radiation to prevent new moles from forming",
                "Avoid picking or scratching at moles to prevent irritation or infection",
            ],
            "precautionary_measures": [
                "Regularly inspect moles for any changes and consult with a dermatologist if concerned",
                "Consider having suspicious moles evaluated for biopsy or removal",
            ],
        },
        "Vascular lesions": {
            "report": "The patient may be presenting with vascular lesions, which include a variety of skin conditions like hemangiomas and telangiectasias. While often harmless, treatment may be desired for cosmetic or symptomatic reasons.",
            "preventative_measures": [
                "Avoid trauma or injury to the skin, which can exacerbate vascular lesions",
                "Protect skin from UV radiation with sunscreen and clothing",
                "Maintain good overall skin health with regular moisturizing and hydration",
            ],
            "precautionary_measures": [
                "Consult with a dermatologist for evaluation and discussion of treatment options",
                "Consider laser therapy or other interventions for cosmetic or symptomatic improvement",
            ],
        },
    }

    # Retrieve medical information based on predicted label
    medical_report = skin_disease_info[predicted_label]["report"]
    preventative_measures = skin_disease_info[predicted_label]["preventative_measures"]
    precautionary_measures = skin_disease_info[predicted_label]["precautionary_measures"]

    # Generate conversational medical report with each section in a paragraphic fashion
    report = (
        "Skin Disease Report:\n\n"
        + medical_report
        + "\n\nPreventative Measures:\n\n- "
        + ",\n- ".join(preventative_measures)
        + "\n\nPrecautionary Measures:\n\n- "
        + ",\n- ".join(precautionary_measures)
    )
    precautions = precautionary_measures

    return report, precautions


def initialize_database(json_file_path="data.json"):
    try:
        # Check if JSON file exists
        if not os.path.exists(json_file_path):
            # Create an empty JSON structure
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")



def save_image(image_file, json_file_path="data.json"):
    try:
        if image_file is None:
            st.warning("No file uploaded.")
            return

        if not session_state["logged_in"] or not session_state["user_info"]:
            st.warning("Please log in before uploading images.")
            return

        # Load user data from JSON file
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        # Find the user's information
        for user_info in data["users"]:
            if user_info["email"] == session_state["user_info"]["email"]:
                image = Image.open(image_file)

                if image.mode == "RGBA":
                    image = image.convert("RGB")

                # Convert image bytes to Base64-encoded string
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="JPEG")
                image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

                # Update the user's information with the Base64-encoded image string
                user_info["skin"] = image_base64

                # Save the updated data to JSON
                with open(json_file_path, "w") as json_file:
                    json.dump(data, json_file, indent=4)

                session_state["user_info"]["skin"] = image_base64
                return

        st.error("User not found.")
    except Exception as e:
        st.error(f"Error saving skin image to JSON: {e}")

def create_account(name, email, age, sex, password, json_file_path="data.json"):
    try:
        # Check if the JSON file exists or is empty
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,
            "report": None,
            "precautions": None,
            "skin":None

        }
        data["users"].append(user_info)

        # Save the updated data to JSON
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None



def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
        else:
            st.error("Invalid credentials. Please try again.")

def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None

def render_dashboard(user_info, json_file_path="data.json"):
    try:
        st.title(f"Welcome to the Dashboard, {user_info['name']}!")
        st.markdown(
    """
    <div style='text-align: left;'>
        <p>There are 7 different classes of skin cancer which are listed below:</p>
        <ul>
            <li style='color: #008000;'>Actinic keratoses</li>
            <li style='color: #ff6600;'>Basal cell carcinoma</li>
            <li style='color: #663300;'>Benign keratosis-like lesions</li>
            <li style='color: #ffcc00;'>Dermatofibroma</li>
            <li style='color: #6600cc;'>Melanoma</li>
            <li style='color: #990000;'>Melanocytic nevi</li>
            <li style='color: #cc0000;'>Vascular lesions</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
        st.subheader("User Information:")
        st.write(f"Name: {user_info['name']}")
        st.write(f"Sex: {user_info['sex']}")
        st.write(f"Age: {user_info['age']}")

        # Open the JSON file and check for the 'skin' key
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == user_info["email"]:
                    if "skin" in user and user["skin"] is not None:
                        image_data = base64.b64decode(user["skin"])
                        st.image(Image.open(io.BytesIO(image_data)), caption="Uploaded Skin Image", use_column_width=True)

                    if isinstance(user_info["precautions"], list):
                        st.subheader("Precautions:")
                        for precautopn in user_info["precautions"]:
                            st.write(precautopn)                    
                    else:
                        st.warning("Reminder: Please upload skin images and generate a report.")
    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")
def fetch_precautions(user_info):
    return (
        user_info["precautions"]
        if user_info["precautions"] is not None
        else "Please upload skin images and generate a report."
    )


def main(json_file_path="data.json"):
    st.sidebar.title("Skin disease prediction system")
    page = st.sidebar.radio(
        "Go to",
        ("Signup/Login", "Dashboard", "Upload Skin Image", "View Reports"),
        key="Skin disease prediction system",
    )

    if page == "Signup/Login":
        st.title("Signup/Login Page")
        login_or_signup = st.radio(
            "Select an option", ("Login", "Signup"), key="login_signup"
        )
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)

    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")

    elif page == "Upload Skin Image":
        st.markdown(
    """
    <div style='text-align: left;'>
        <p>There are 7 different classes of skin cancer which are listed below:</p>
        <ul>
            <li style='color: #008000;'>Actinic keratoses</li>
            <li style='color: #ff6600;'>Basal cell carcinoma</li>
            <li style='color: #663300;'>Benign keratosis-like lesions</li>
            <li style='color: #ffcc00;'>Dermatofibroma</li>
            <li style='color: #6600cc;'>Melanoma</li>
            <li style='color: #990000;'>Melanocytic nevi</li>
            <li style='color: #cc0000;'>Vascular lesions</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
        if session_state.get("logged_in"):
            st.title("Upload Skin Image")
            uploaded_image = st.file_uploader(
                "Choose a skin image (JPEG/PNG)", type=["jpg", "jpeg", "png"]
            )
            if st.button("Upload") and uploaded_image is not None:
                st.image(uploaded_image, use_column_width=True)
                st.success("Skin image uploaded successfully!")
                model = load_model("Skin_disease_ann1.h5")
                prediction, confidence, img_array = get_predictions(uploaded_image,model)

                classes = {
                    0: "Actinic keratoses",
                    1: "Basal cell carcinoma",
                    2: "Benign keratosis-like lesions",
                    3: "Dermatofibroma",
                    4: "Melanoma",
                    5: "Melanocytic nevi",
                    6: "Vascular lesions"
                }

                if prediction==0:
                    st.markdown(f"<h2 style='text-align: center; color: red; font: Times New Roman;'>Actinic keratoses </h2>", unsafe_allow_html=True)
                elif prediction==1:
                    st.markdown(f"<h2 style='text-align: center; color: red; font: Times New Roman;'>Basal cell carcinoma</h2>", unsafe_allow_html=True)
                elif prediction==2:
                    st.markdown(f"<h2 style='text-align: center; color: red; font: Times New Roman;'>Benign keratosis-like lesions</h2>", unsafe_allow_html=True)
                elif prediction==3:
                    st.markdown(f"<h2 style='text-align: center; color: red; font: Times New Roman;'>Dermatofibroma</h2>", unsafe_allow_html=True)
                elif prediction==4:
                    st.markdown(f"<h2 style='text-align: center; color: red; font: Times New Roman;'>Melanoma</h2>", unsafe_allow_html=True)
                elif prediction==5:
                    st.markdown(f"<h2 style='text-align: center; color: red; font: Times New Roman;'>Melanocytic nevi</h2>", unsafe_allow_html=True)
                elif prediction==6:
                    st.markdown(f"<h2 style='text-align: center; color: red; font: Times New Roman;'>Vascular lesions</h2>", unsafe_allow_html=True)

                st.markdown(
                    f"<p style='text-align: center; color: #ff6600; font-size: 1.5em;'><b>Confidence:</b> {confidence:.2%}</p>",
                    unsafe_allow_html=True)
                

                background_batch = load_background_batch()
                shap_values = shap_explanation(model, img_array, background_batch) 
                show_shap(shap_values, img_array, prediction)
                
                save_image(uploaded_image, json_file_path)

                model = load_model("Skin_disease_ann1.h5")
                condition = predict_skin(uploaded_image, model)
                report, precautions = generate_medical_report(condition)

                # Read the JSON file, update user info, and write back to the file
                with open(json_file_path, "r+") as json_file:
                    data = json.load(json_file)
                    user_index = next((i for i, user in enumerate(data["users"]) if user["email"] == session_state["user_info"]["email"]), None)
                    if user_index is not None:
                        user_info = data["users"][user_index]
                        user_info["report"] = report
                        user_info["precautions"] = precautions
                        session_state["user_info"] = user_info
                        json_file.seek(0)
                        json.dump(data, json_file, indent=4)
                        json_file.truncate()
                    else:
                        st.error("User not found.")
                st.write(report)
        else:
            st.warning("Please login/signup to upload a skin image.")

    elif page == "View Reports":
        if session_state.get("logged_in"):
            st.title("View Reports")
            user_info = get_user_info(session_state["user_info"]["email"], json_file_path)
            if user_info is not None:
                if user_info["report"] is not None:
                    st.subheader("Skin Report:")
                    st.write(user_info["report"])
                else:
                    st.warning("No reports available.")
            else:
                st.warning("User information not found.")
        else:
            st.warning("Please login/signup to view reports.")

if __name__ == "__main__":
    initialize_database()
    main()
