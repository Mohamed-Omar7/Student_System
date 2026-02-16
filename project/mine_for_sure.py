import tkinter as tk
from tkinter import filedialog, messagebox
import face_recognition as fr
import os, pickle, cv2
import numpy as np
from sklearn.cluster import DBSCAN

# ------------------------------
# CONFIGURATION
# ------------------------------
KNOWN_DIR   = "known_faces"        # folder for labeled images
CLUSTER_PKL = "cluster_fi.pkl"     # file to save clusters

# ------------------------------
# FACE RECOGNITION LOGIC
# ------------------------------

def recognize_face(upload_path):
    print(f"Starting recognition for: {upload_path}")

    # 1. Load known faces
    known_face_encodings = []
    known_face_names     = []

    if not os.path.exists(KNOWN_DIR):
        os.makedirs(KNOWN_DIR)

    for filename in os.listdir(KNOWN_DIR):
        if filename.lower().endswith((".jpg", ".png")):
            img  = fr.load_image_file(os.path.join(KNOWN_DIR, filename))
            encs = fr.face_encodings(img)
            if encs:
                known_face_encodings.append(encs[0])
                known_face_names.append(os.path.splitext(filename)[0])

    # 2. Load uploaded face
    unknown_image = fr.load_image_file(upload_path)
    unknown_encs  = fr.face_encodings(unknown_image)

    if not unknown_encs:
        messagebox.showerror("Error", "No face detected in uploaded photo.")
        return

    unknown_enc = unknown_encs[0]

    # 3. Try normal 1‑NN identification
    matches   = fr.compare_faces(known_face_encodings, unknown_enc, tolerance=0.55)
    distances = fr.face_distance(known_face_encodings, unknown_enc)

    if True in matches:
        best_idx = distances.argmin()
        messagebox.showinfo("Match Found", f"Match found: {known_face_names[best_idx]}")
        return

    # 4. Clustering fallback
    if os.path.exists(CLUSTER_PKL):
        with open(CLUSTER_PKL, "rb") as f:
            cluster_data = pickle.load(f)
    else:
        cluster_data = {}

    cluster_encs = [enc for encs in cluster_data.values() for enc in encs]
    all_encs = np.vstack([cluster_encs, unknown_enc]) if cluster_encs else np.array([unknown_enc])
    labels   = DBSCAN(eps=0.5, min_samples=2, metric="euclidean").fit(all_encs).labels_

    new_face_label = labels[-1]

    if new_face_label == -1:
        new_face_label = max(labels[:-1], default=-1) + 1
        cluster_data[new_face_label] = [unknown_enc]
        messagebox.showinfo("New Face", f"Unknown person. Started new cluster #{new_face_label}.")
    else:
        cluster_data.setdefault(new_face_label, []).append(unknown_enc)
        messagebox.showinfo("Cluster Update", f"Added to existing cluster #{new_face_label} "
                             f"(size = {len(cluster_data[new_face_label])}).")

    # 5. Save cluster updates
    with open(CLUSTER_PKL, "wb") as f:
        pickle.dump(cluster_data, f)

    # 6. Option to name the cluster (if only one image inside)
    if len(cluster_data.get(new_face_label, [])) == 1:
        name = simple_input_dialog(f"Enter a name for this person (or leave blank to skip):")
        if name:
            dest = os.path.join(KNOWN_DIR, name.replace(' ', '_') + ".jpg")
            cv2.imwrite(dest, cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Saved", f"Saved face to '{dest}'. Future runs will identify as {name}.")
            known_face_encodings.append(unknown_enc)
            known_face_names.append(name)
            cluster_data.pop(new_face_label)
            with open(CLUSTER_PKL, "wb") as f:
                pickle.dump(cluster_data, f)

# ------------------------------
# UI SETUP
# ------------------------------

def browse_file():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        selected_file.set(file_path)

def start_recognition():
    path = selected_file.get()
    if not path:
        messagebox.showerror("Error", "Please select an image first!")
        return
    recognize_face(path)

def simple_input_dialog(prompt_text):
    # very basic text input popup
    input_window = tk.Toplevel(root)
    input_window.title("Input Name")

    tk.Label(input_window, text=prompt_text).pack(pady=10)
    entry = tk.Entry(input_window, width=40)
    entry.pack(pady=5)

    def submit():
        user_input.set(entry.get())
        input_window.destroy()

    submit_btn = tk.Button(input_window, text="Submit", command=submit)
    submit_btn.pack(pady=10)

    input_window.transient(root)
    input_window.grab_set()
    root.wait_window(input_window)
    return user_input.get()

# ------------------------------
# MAIN WINDOW
# ------------------------------

root = tk.Tk()
root.title("Face Recognition App")
root.geometry("400x250")

selected_file = tk.StringVar()
user_input    = tk.StringVar()

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

browse_btn = tk.Button(frame, text="Browse Image", command=browse_file, height=2, width=20)
browse_btn.pack(pady=10)

path_label = tk.Label(frame, textvariable=selected_file, wraplength=300)
path_label.pack(pady=5)

recognize_btn = tk.Button(frame, text="Recognize Face", command=start_recognition, height=2, width=20)
recognize_btn.pack(pady=10)

root.mainloop()