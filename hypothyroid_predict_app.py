import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, \
    QDialog, QTextEdit, QMessageBox, QMenu, QMenuBar, QLineEdit, QComboBox
from PyQt6.QtGui import QPixmap, QAction, QDoubleValidator
from PyQt6.QtCore import Qt, QSize, QUrl
from PyQt6.QtGui import QDesktopServices
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('realistic_normalized_synthetic_thyroid_data.csv')

# Separate features and target variable
X = df[['age', 'sex', 'pregnant', 'goitre', 'tumor', 'TSH', 'T3', 'TT4']]
y = df['Class']

# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

class HypothyroidPredictApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setFixedSize(1100, 600)  # Set the window size to width 1100 and height 600

        self.setWindowTitle("Hypothyroid Prediction App")
        self.setGeometry(100, 100, 1100, 600)
        # self.setStyleSheet("background-color: white;")  # Set white background

        self.create_menu()  # Call the create_menu function

        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Main layout
        main_layout = QHBoxLayout(main_widget)

        # Add left frame
        left_frame = self.create_left_frame()
        main_layout.addWidget(left_frame)

        # Add right frame
        right_frame = self.create_right_frame()
        main_layout.addWidget(right_frame)

        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 2)

    def create_menu(self):
        menubar = QMenuBar(self)

        # File Menu
        file_menu = QMenu("File", self)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        menubar.addMenu(file_menu)

        # About Menu
        about_menu = QMenu("About", self)
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_info)
        about_menu.addAction(about_action)
        menubar.addMenu(about_menu)

        self.setMenuBar(menubar)

    def show_about_info(self):
        QMessageBox.information(self, "About",
            "This software is created by Joel and Lucy 2024. It is a development app for project work purposes.\nVersion 1.0 2024")

    def create_left_frame(self):
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.Shape.Box)
        left_frame.setFrameShadow(QFrame.Shadow.Raised)
        left_frame.setStyleSheet("border-radius: 15px; background-color: white;")

        left_layout = QVBoxLayout(left_frame)

        # Add image frame
        image1_frame = self.create_image_frame("hypotyroid.png")
        left_layout.addWidget(image1_frame)

        # Add buttons frame
        buttons1_frame = self.create_buttons1_frame()
        left_layout.addWidget(buttons1_frame)

        return left_frame

    def create_image_frame(self, image_path):
        image_frame = QFrame()
        image_frame.setFrameShape(QFrame.Shape.Box)
        image_frame.setFrameShadow(QFrame.Shadow.Raised)
        image_frame.setStyleSheet("""
            border-radius: 15px;
            background-color: white;
            border: 2px solid white;
        """)  # Removed border line

        image_label = QLabel(image_frame)
        pixmap = QPixmap(image_path)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout = QVBoxLayout(image_frame)
        image_layout.addWidget(image_label)

        return image_frame

    def create_buttons1_frame(self):
        buttons1_frame = QFrame()
        buttons1_frame.setFrameShape(QFrame.Shape.Box)
        buttons1_frame.setFrameShadow(QFrame.Shadow.Raised)
        buttons1_frame.setStyleSheet("border-radius: 15px; background-color: white;")

        predict_button = QPushButton("PREDICT")
        more_info_button = QPushButton("More INFO")

        buttons1_layout = QHBoxLayout(buttons1_frame)
        buttons1_layout.addWidget(predict_button)
        buttons1_layout.addWidget(more_info_button)

        button_style = """
        QPushButton {
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            color: white;
            background-color: #3498db; /* Default color */
        }
        QPushButton:hover {
            background-color: #e74c3c; /* Red color on hover */
        }
        """

        predict_button.setObjectName("predict_button")
        more_info_button.setObjectName("more_info_button")

        predict_button.setStyleSheet(button_style)
        more_info_button.setStyleSheet(button_style)

        predict_button.clicked.connect(self.open_predict_window)  # Connect to the predict window

        # Connect more_info_button to open Cleveland Clinic website
        more_info_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://my.clevelandclinic.org/health/diseases/12120-hypothyroidism")))

        return buttons1_frame


    def create_right_frame(self):
        right_frame = QFrame()
        right_frame.setFrameShape(QFrame.Shape.Box)
        right_frame.setFrameShadow(QFrame.Shadow.Raised)
        right_frame.setStyleSheet("border-radius: 15px; background-color: white;")

        right_layout = QVBoxLayout(right_frame)

        # Add buttons frame
        buttons2_frame = self.create_buttons2_frame()
        right_layout.addWidget(buttons2_frame)

        # Add image frame with resized and rounded image
        image2_frame = self.create_image2_frame("care2.jpg", QSize(500, 500))  # Resize the image
        right_layout.addWidget(image2_frame)

        right_layout.setStretch(0, 1)
        right_layout.setStretch(1, 2)

        return right_frame

    def create_image2_frame(self, image_path, size):
        image_frame = QFrame()
        image_frame.setFrameShape(QFrame.Shape.Box)
        image_frame.setFrameShadow(QFrame.Shadow.Raised)
        image_frame.setStyleSheet("""
            border-radius: 15px;
            background-color: white;
            border: 2px solid white;
        """)

        image_label = QLabel(image_frame)
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setFixedSize(size)

        image_layout = QVBoxLayout(image_frame)
        image_layout.addWidget(image_label)

        return image_frame

    def create_buttons2_frame(self):
        buttons2_frame = QFrame()
        buttons2_frame.setFrameShape(QFrame.Shape.Box)
        buttons2_frame.setFrameShadow(QFrame.Shadow.Raised)
        buttons2_frame.setStyleSheet("border-radius: 15px; background-color: white;")

        contact_button = QPushButton("CONTACT")
        watch_button = QPushButton("WATCH")

        contact_button.clicked.connect(self.show_contact_info)
        watch_button.clicked.connect(self.open_watch_link)

        buttons2_layout = QHBoxLayout(buttons2_frame)
        buttons2_layout.addWidget(contact_button)
        buttons2_layout.addWidget(watch_button)

        button_style = """
        QPushButton {
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            color: white;
            background-color: #3498db; /* Default color */
        }
        QPushButton:hover {
            background-color: #e74c3c; /* Red color on hover */
        }
        """

        contact_button.setObjectName("contact_button")
        watch_button.setObjectName("watch_button")

        contact_button.setStyleSheet(button_style)
        watch_button.setStyleSheet(button_style)

        return buttons2_frame

    def show_contact_info(self):
        contact_info = """
        Care Provider: +1234567890
        Email: contact@hypothyroidapp.com
        Website: www.hypothyroidapp.com
        Facebook: facebook.com/hypothyroidapp
        Twitter: twitter.com/hypothyroidapp
        Instagram: instagram.com/hypothyroidapp
        LinkedIn: linkedin.com/company/hypothyroidapp
        """

        dialog = QDialog(self)
        dialog.setWindowTitle("Contact Information")
        dialog.setGeometry(300, 300, 400, 200)

        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit(dialog)
        text_edit.setReadOnly(True)
        text_edit.setText(contact_info)
        layout.addWidget(text_edit)

        dialog.exec()

    def open_watch_link(self):
        QDesktopServices.openUrl(QUrl("https://www.youtube.com/watch?v=hLNXJWLsjAE&pp=ygUOaHlwb3RoeXJvaWRpc20%3D"))

    def open_predict_window(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Predict")
        dialog.setGeometry(300, 300, 400, 400)

        layout = QVBoxLayout(dialog)

        age_label = QLabel("Age:")
        self.age_input = QLineEdit()
        layout.addWidget(age_label)
        layout.addWidget(self.age_input)

        sex_label = QLabel("Sex:")
        self.sex_input = QComboBox()
        self.sex_input.addItems(["Male", "Female"])
        layout.addWidget(sex_label)
        layout.addWidget(self.sex_input)

        pregnant_label = QLabel("Pregnant:")
        self.pregnant_input = QComboBox()
        self.pregnant_input.addItems(["YES", "NO"])
        layout.addWidget(pregnant_label)
        layout.addWidget(self.pregnant_input)

        goitre_label = QLabel("Goitre:")
        self.goitre_input = QComboBox()
        self.goitre_input.addItems(["YES", "NO"])
        layout.addWidget(goitre_label)
        layout.addWidget(self.goitre_input)

        tumor_label = QLabel("Tumor:")
        self.tumor_input = QComboBox()
        self.tumor_input.addItems(["YES", "NO"])
        layout.addWidget(tumor_label)
        layout.addWidget(self.tumor_input)

        tsh_label = QLabel("Thyroid Stimulating Hormone (TSH):")
        self.tsh_input = QLineEdit()
        self.tsh_input.setPlaceholderText("Normal range: 0.5 - 5.0 mU/L")
        layout.addWidget(tsh_label)
        layout.addWidget(self.tsh_input)

        t3_label = QLabel("Triiodothyronine (T3):")
        self.t3_input = QLineEdit()
        self.t3_input.setPlaceholderText("Normal range: 0.8 - 2.0 ng/mL")
        layout.addWidget(t3_label)
        layout.addWidget(self.t3_input)

        tt4_label = QLabel("Total Thyroxine (TT4):")
        self.tt4_input = QLineEdit()
        self.tt4_input.setPlaceholderText("Normal range: 4.5 - 12.0 µg/dL")
        layout.addWidget(tt4_label)
        layout.addWidget(self.tt4_input)

        predict_now_button = QPushButton("Predict Now")
        predict_now_button.clicked.connect(self.predict_now)
        layout.addWidget(predict_now_button)

        dialog.exec()

    def predict_now(self):
        age = self.age_input.text()
        sex = self.sex_input.currentText().lower()
        pregnant = self.pregnant_input.currentText().lower()
        goitre = self.goitre_input.currentText().lower()
        tumor = self.tumor_input.currentText().lower()
        tsh = self.tsh_input.text()
        t3 = self.t3_input.text()
        tt4 = self.tt4_input.text()

        # Validate inputs
        try:
            age = float(age)
            tsh = float(tsh)
            t3 = float(t3)
            tt4 = float(tt4)
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numerical values for age, TSH, T3, and TT4.")
            return

        # Convert categorical inputs to numerical values
        sex = 1 if sex == 'male' else 0
        pregnant = 1 if pregnant == 'yes' else 0
        goitre = 1 if goitre == 'yes' else 0
        tumor = 1 if tumor == 'yes' else 0

        # Create a dataframe for the input
        input_data = pd.DataFrame([[age, sex, pregnant, goitre, tumor, tsh, t3, tt4]],
                                  columns=['age', 'sex', 'pregnant', 'goitre', 'tumor', 'TSH', 'T3', 'TT4'])

        # Make the prediction
        prediction = clf.predict(input_data)[0]

        # Generate detailed recommendation
        if prediction == 1:
            result_message = "The prediction is Positive for Hypothyroidism."
            recommendation = ("It is recommended to consult with a healthcare provider for a thorough evaluation. "
                              "You may need further tests and a detailed assessment of your thyroid function. "
                              "Maintaining a balanced diet, managing stress, and following your doctor's advice is crucial.")
        else:
            result_message = "The prediction is Negative for Hypothyroidism."
            recommendation = ("Your results do not suggest hypothyroidism at this time. "
                              "However, if you have symptoms or concerns, it's a good idea to consult with a healthcare provider. "
                              "Regular check-ups and a healthy lifestyle are always beneficial.")

        # Display the result
        QMessageBox.information(self, "Prediction Result", f"{result_message}\n\n{recommendation}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HypothyroidPredictApp()
    window.show()
    sys.exit(app.exec())
