import face_recognition
import os
import numpy as np
from tkinter import Tk, filedialog, Label, Button, messagebox, StringVar, OptionMenu #UI
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pyttsx3 # voice

class StudentSystem:
    def __init__(self):  # <-- Correct constructor name
        # Initialize both systems
        self.load_known_faces()
        self.load_student_data()
        
        # Create main application window
        self.root = Tk()
        self.root.title("Integrated Student System")
        self.root.geometry("800x600")
        
        # Face Recognition Section
        self.face_recognition_frame = Label(self.root, text="Face Recognition", font=("Arial", 19))
        self.face_recognition_frame.pack(pady=10)
        
        self.upload_btn = Button(self.root, text="Upload Image for Recognition", 
                                command=self.upload_image, height=2, width=30,bg="purple",font=("Arial",19))
        self.upload_btn.pack()
        
        self.face_result_label = Label(self.root, text="", font=("Arial", 12))
        self.face_result_label.pack(pady=10)
        
        # Grade Prediction Section
        self.grade_frame = Label(self.root, text="Grade Prediction", font=("Arial", 14))
        self.grade_frame.pack(pady=10)
        
        # Subject dropdown

        self.subject_var = StringVar(self.root)
        self.subject_var.set("Select Subject")
        self.subject_menu = OptionMenu(self.root, self.subject_var, *self.available_subjects) # (*) to make a seprated list
        self.subject_menu.config(width=20)
        self.subject_menu.pack(pady=5)
        
        # Student dropdown
        self.student_var = StringVar(self.root)
        self.student_var.set("Select Student")
        self.student_menu = OptionMenu(self.root, self.student_var, *self.available_students)
        self.student_menu.config(width=20)
        self.student_menu.pack(pady=5)
        
        self.predict_btn = Button(self.root, text="Predict Grade", 
                                command=self.predict_grade, height=2, width=20)
        self.predict_btn.pack(pady=10)
        
        self.grade_result_label = Label(self.root, text="", font=("Arial", 12))
        self.grade_result_label.pack(pady=10)
        
    def load_known_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []
        
        if not os.path.exists("known_faces"):
            os.makedirs("known_faces")
            messagebox.showinfo("Info", "Created 'known_faces' folder. Please add student images there.")
            return
            
        for filename in os.listdir("known_faces"):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join("known_faces", filename)
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(os.path.splitext(filename)[0])
    
    def load_student_data(self):
        try:
            self.data = pd.read_excel('d:\\al_test\\pythonXvs\\project\\students.xlsx')
            self.available_subjects = sorted(self.data["Subject"].unique())
            self.available_students = sorted(self.data["Student Name"].unique())
        except FileNotFoundError:
            messagebox.showerror("Error", "students.xlsx file not found!")
            self.root.destroy()
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".jpg;.png;*.jpeg")])
        if not file_path:
            return
            
        unknown_image = face_recognition.load_image_file(file_path)
        unknown_encodings = face_recognition.face_encodings(unknown_image)
        
        if unknown_encodings:
            unknown_encoding = unknown_encodings[0]
            results = face_recognition.compare_faces(self.known_face_encodings, unknown_encoding)
            distances = face_recognition.face_distance(self.known_face_encodings, unknown_encoding)
            
            if True in results:
                best_match_index = np.argmin(distances)
                name = self.known_face_names[best_match_index]
                self.face_result_label.config(text=f"Match found: {name}")
                self.student_var.set(name)  # Auto-fill the student name
            else:
                self.face_result_label.config(text="No match found.")
        else:
            self.face_result_label.config(text="No face detected.")
    
    def predict_grade(self):
        subject_name = self.subject_var.get()
        student_name = self.student_var.get()
        
        if subject_name == "Select Subject" or student_name == "Select Student":
            messagebox.showwarning("Warning", "Please select both subject and student!")
            return
            
        if subject_name in self.data["Subject"].values and student_name in self.data["Student Name"].values:
            student_subject_data = self.data[(self.data['Subject'] == subject_name) & 
                                           (self.data['Student Name'] == student_name)]
            
            if len(student_subject_data) > 0:
                X = np.array(range(len(student_subject_data))).reshape(-1, 1)
                y = student_subject_data['Grade'].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                next_subject = np.array([[len(student_subject_data)]])
                predicted_score = model.predict(next_subject)
                
                result_text = f"Predicted score for {student_name} in {subject_name}: {predicted_score[0]:.2f}"
                self.grade_result_label.config(text=result_text)
                
                # Plotting
                plt.figure(figsize=(10, 5))
                plt.scatter(X, y, color='blue', label='Actual Grades')
                plt.plot(X, model.predict(X), color='red', label='Prediction Line')
                plt.scatter(len(student_subject_data), predicted_score[0], color='green', 
                           marker='*', s=200, label=f'Predicted: {predicted_score[0]:.1f}')
                plt.xlabel('Exam Number')
                plt.ylabel('Grade')
                plt.title(f'Grade Prediction for {student_name} in {subject_name}')
                plt.legend()
                plt.show()
                
                # Voice output
                engine = pyttsx3.init()
                engine.say(result_text)
                engine.runAndWait()
            else:
                messagebox.showinfo("Info", f"No grades found for {student_name} in {subject_name}")
        else:
            messagebox.showerror("Error", "Invalid subject or student selection!")
    
    def run(self):
        self.root.mainloop() 

# Run the application
if __name__ == "__main__":  # <-- Correct way to check
    app = StudentSystem()
    app.run()