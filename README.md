# pyeem
Python Library for Processing EEM Spectra including training and vizualization of convolutional neural networks to identify chemicals or components present in samples. 

## pyeem functions 
#### (under development)
* Import meta-data
* Import Spectra from instument file(s)
   - EEMs
   - Absorbance Measurements
* Subtract blanks
* Remove Scatter
  - Using interpolation
  - Replace with fixed value (i.e. zero)
* Ramam normalize
* Crop EEM
* Inner filter effect correction
* **EEM Vizualization**
  - Contour plots
* **Convolutional Neural Network (CNN)**
  - Data augmentation
  - Network Training
  - Result Vizualization (Parity Plots)
  - Salinecy Maps and Vizualizations