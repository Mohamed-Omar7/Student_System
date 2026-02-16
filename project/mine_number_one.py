import face_recognition as fr
import os, pickle, cv2
import numpy as np
from sklearn.cluster import DBSCAN

KNOWN_DIR   = "knwww.xlsx"        # folder with labeled images
CLUSTER_PKL = "cluster_fi.pkl"     # where clusters live
UPLOAD_PATH = "messi.png"            # image you’re processing

# --------------------------------------------------
# 1. Load / encode known faces
# --------------------------------------------------
known_face_encodings = []
known_face_names     = []
 
for filename in os.listdir(KNOWN_DIR):
    if filename.lower().endswith((".jpg", ".png")):
        img  = fr.load_image_file(os.path.join(KNOWN_DIR, filename))
        encs = fr.face_encodings(img)
        if encs:
            known_face_encodings.append(encs[0])
            known_face_names.append(os.path.splitext(filename)[0])

# --------------------------------------------------
# 2. Load / encode the uploaded face
# --------------------------------------------------
unknown_image = fr.load_image_file(UPLOAD_PATH)
unknown_encs  = fr.face_encodings(unknown_image)

if not unknown_encs:
    print("No face detected in uploaded photo.")
    quit()

unknown_enc = unknown_encs[0]

# --------------------------------------------------
# 3. Try normal 1‑NN identification
# --------------------------------------------------
matches   = fr.compare_faces(known_face_encodings, unknown_enc, tolerance=0.55)
distances = fr.face_distance(known_face_encodings, unknown_enc)

if True in matches:
    best_idx = distances.argmin()
    print(f"Match found: {known_face_names[best_idx]}")
    quit()

print("No match in labeled gallery. Switching to clustering…")

# --------------------------------------------------
# 4. Load previous cluster encodings (if any)
# --------------------------------------------------
if os.path.exists(CLUSTER_PKL):
    with open(CLUSTER_PKL, "rb") as f:
        cluster_data = pickle.load(f)      # {label: [encodings]}
else:
    cluster_data = {}

# Flatten existing cluster encodings into one array
cluster_encs = [enc for encs in cluster_data.values() for enc in encs]

# Append the new unknown face and cluster everything
all_encs = np.vstack([cluster_encs, unknown_enc]) if cluster_encs else np.array([unknown_enc])
labels   = DBSCAN(eps=0.5, min_samples=2, metric="euclidean").fit(all_encs).labels_

new_face_label = labels[-1]                 # label for uploaded face

if new_face_label == -1:                    # create new cluster
    new_face_label = max(labels[:-1], default=-1) + 1
    cluster_data[new_face_label] = [unknown_enc]
    print(f"Unknown person. Started new cluster #{new_face_label}.")
else:                                       # join existing
    cluster_data.setdefault(new_face_label, []).append(unknown_enc)
    print(f"Added to existing cluster #{new_face_label} "
          f"(size = {len(cluster_data[new_face_label])}).")

# --------------------------------------------------
# 5. Prompt to name single‑face clusters & auto‑save
# --------------------------------------------------
if len(cluster_data[new_face_label]) == 1:   # first time seeing this face
    name = input(f"Enter a name for cluster #{new_face_label} "
                 "(or press Enter to skip): ").strip()
    if name:
        # save current upload image to known_faces/
        dest = os.path.join(KNOWN_DIR, name.replace(' ', '_') + ".jpg")
        cv2.imwrite(dest, cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR))
        print(f"Saved face to '{dest}'. Future runs will identify them as {name}.")
        # promote to known list
        known_face_encodings.append(unknown_enc)
        known_face_names.append(name)
        # remove cluster entry
        cluster_data.pop(new_face_label)

# --------------------------------------------------
# 6. Save updated clusters
# --------------------------------------------------
with open(CLUSTER_PKL, "wb") as f:
    pickle.dump(cluster_data, f)