from sklearn.datasets import fetch_lfw_people
from pathlib import Path
import cv2
import numpy as np
import shutil

print("Loading LFW dataset...")

d = fetch_lfw_people(
    min_faces_per_person=10,
    resize=1.0,
    color=True,
    download_if_missing=True
)

base = Path("data")

shutil.rmtree(base / "enrolled", ignore_errors=True)
shutil.rmtree(base / "evaluation", ignore_errors=True)

(base / "enrolled").mkdir(parents=True)
(base / "evaluation" / "known").mkdir(parents=True)
(base / "evaluation" / "unknown").mkdir(parents=True)

people = [
    i for i in range(len(d.target_names))
    if np.sum(d.target == i) >= 10
]

enrolled = people[:6]
unknown_people = people[6:10]

def save_image(path, image):
    image = cv2.cvtColor(
        (image * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )

    image = cv2.resize(
        image,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    cv2.imwrite(str(path), image)


for n, person in enumerate(enrolled, 1):

    indices = np.where(d.target == person)[0]

    enroll_dir = base / "enrolled" / f"person_{n:02d}"
    known_dir = base / "evaluation" / "known" / f"person_{n:02d}"

    enroll_dir.mkdir(parents=True)
    known_dir.mkdir(parents=True)

    for j, idx in enumerate(indices[:5], 1):
        save_image(
            enroll_dir / f"{j:02d}.jpg",
            d.images[idx]
        )

    for j, idx in enumerate(indices[5:], 1):
        save_image(
            known_dir / f"{j:02d}.jpg",
            d.images[idx]
        )


for n, person in enumerate(unknown_people, 1):

    indices = np.where(d.target == person)[0]

    for j, idx in enumerate(indices[:5], 1):

        save_image(
            base / "evaluation" / "unknown" /
            f"stranger_{n:02d}_{j:02d}.jpg",
            d.images[idx]
        )


print()
print("================================")
print("DATASET REBUILT SUCCESSFULLY")
print("================================")
print("Enrolled identities:", len(enrolled))
print("Unknown identities:", len(unknown_people))
