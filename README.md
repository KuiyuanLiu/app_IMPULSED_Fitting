# IMPULSED GUI 
---------------------------------------------------------------------------------------------------------------
This is a MATLAB application that includes microstructural analysis code developed by Zhejiang University College of Biomedical Engineering and Instrument Science.

Authors:  Dan Wu, Kuiyuan Liu, Ruicheng Ba 
---------------------------------------------------------------------------------------------------------------
# Installation 
1. Run "IMPULSED_GUI_Installer_web.exe". Just follow to select installation path and MATLAB Runtime path.
2. Move the mricron folder to the D drive, (change the mricon folder path to D:\mricron). Make sure the dcm2nii.exe, dcm2niigui.ini, flirt.exe files in the folder are not lost. They are used for dicom file conversion, and image registration, respectively

# Contents
The cell microstructure analysis software based on the diffusion magnetic resonance two-compartment model relies on the object-oriented visualization development tool MATLAB (R2020b) APP Designer of The MathWorks Company to undertake the design of the user visualization interface. 

The software supports the multi-shell acquisition mode of one PGSE and two OGSE. It can convert the data in dicom format to data in nifti format, and register the multi-b-value images in each sequence with the b0 images in PGSE, and then realize the parameter fitting of the two-compartment model according to the input diffusion sequence parameters in the specified ROI，The software can be used to generate the basic apparent diffusion coefficient (ADC), and the two-compartment model-based parameters such as cellularity, extracellular diffusion coefficient, cell diameter, and intracellular fraction.And supports batch processing of files.

The user interface is as shown below
<div align=center><img src="https://github.com/KuiyuanLiu/app_IMPULSED_Fitting/blob/main/readmeImg/layout.png" ></div>

You can learn to use the software by working with the examples we provide.

## Work flow
1. Select the path of the folder to be processed, the data of multiple cases is saved in this folder, and the data of each case is saved in its own folder.DICOM files and ROI files (saved as mask.nii) need to be included in the folder of a single case.
The file is saved in the form as shown in the figure below, and a box in the figure represents a folder.

<div align=center><img src="https://github.com/KuiyuanLiu/app_IMPULSED_Fitting/blob/main/readmeImg/FileFormat.png" ></div>
  
Note that there are three folders in the DICOM folder, and the x, y, and z in each folder name need to be consistent with the input.     That is, the folder name for saving PGSE needs to contain the "bigdelta (user input) MS" field, and the OGSE folder name needs to contain the "f (user input) HZ" field.

After selecting the target file, the software will convert the file format from dicom to nifti.

ROI is saved in mask.nii format


2. Fill in the acquisition parameters for PGSE and OGSE in the boxes. The View b value button on the right can be used to view the b value information read from the dicom file.

3. When fitting the IMPULSED model, the intracellular diffusion coefficient needs to be fixed, which can be specified by the user, the default is 1 μm²/ms.

4. The registration switch can choose whether to register or not.

5. Finally, click the run button to get the fitted parameter image.

## Output
1. The software will generate an image registered to b=0 under the PGSE sequence and analyze the cellularity , intracellular component (unitless decimals 0~1), extracellular diffusion coefficient (μm²/ms), cell Diameter (μm) and apparent diffusion coefficient (μm²/ms) for the three sequences. The analysis results are stored in the "map" folder in the case folder.

2. The registration results are stored in the "reg_temp" folder in the case folder according to the different b values. The beginning of dwi is the original image of different b values, and the beginning of flirt_dwi is the result of registration of different b values to PGSE b=0. After integration, it is stored in the case folder, named as "data_alignged_all.nii".

# Comments or questions? 
Please send your comments or questions to Kuiyuan Liu (kuiyuanliu@zju.edu.cn) 
