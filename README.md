# 29_AI_Final_Project

## Title: Attendance Registration System from Fingerprint Details and Gender.

### Link to Presentation:
[Youtube](https://youtu.be/i8h-OAiMg3M)

Fingerprints can be used in all sorts of ways:
- Providing biometric security (for example, to control access to secure areas or systems)
- Identifying amnesia victims and unknown deceased (such as victims of major disasters, if their fingerprints are on file)
- Conducting background checks (including applications for government employment, defense security clearance, concealed weapon permits, etc.).

This repository processes an attendance registration system, whereby the person's gender is the underlying distinguishing model for their classification. This breakdown decreases the time it takes for looping over the dataset to compare and acquire fingerprint similarity with those in the database. An image classification model is built as the underlying model to separate the images by gender.

### Prerequisites

- Python
- OpenCV

### Algorithm Pipeline

The techniques of fingerprint recognition that we used take the fingerprint and classify it into a male or a female. With this, it determines the folder to go into. When it enters the folder, It uses the similarity model to determine if the image is in that folder. When it enters the folder, it will segment the ridge region, morphological thinning, and estimate the local orientation of ridges in a fingerprint.

The CV2 uses the BF Matcher and the KNN matcher to match the images based on points of high similarity.

The images for comparison are placed in two files, for males and for females.

After the classification and similarity test, the accuracy level is derived, and the person's name is displayed.

When the name is displayed, it is also printed on another file, which, after all the persons who have to check their identification are done, can then be printed or sent to the lecturer.

Because our dataset had both altered and original images, we used the altered images for some test, to ensure even if a live image was taken which is not identical to the original image, it would be displayed. This was proven to be true.

### DATA

The data used for this project can be found [in the link](#), with focus on only the real data.

### Acknowledgements

This project is based on papers like the SOCOFIng FINAL Research paper:
Ozkaya, N., & Sagiroglu, S. (2010). Generating one biometric feature from another: Faces from fingerprints. Sensors (Basel, Switzerland). [Read more](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3292116/)

### Conclusion

This project focused on establishing the relationship between fingerprints and gender of people, which was established. This is a foundation for the advancement of research already being carried out in the biometric field.

### References

- [Fingerprint Matching in Python](https://www.youtube.com/watch?v=IIvfqfKkiio&pp=ygUYU09DT2ZpbmcgaWRlbnRpZmljYXRpb24_)
- [Build a deep CNN model with any image](https://youtu.be/jztwpsIzEGc) by Nicolas Renotte.
