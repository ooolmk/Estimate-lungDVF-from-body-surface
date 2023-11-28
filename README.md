# Estimate-lungDVF-from-body-surface
Real-time estimation of lung deformation from body surface using a general CoordConv CNN.

Note the path parameters in the replacement code.

for 4D-lung dataset
first run data_clean.py to remove 4D-CBCT use only 4D-FBCT for training. And turn .dcm into .npy form.
then turn all the .npy into .mat form and run ptv_4dlung.m.

for dir-lab dataset
Replace the DIR_test_all.m in ptvreg project (https://github.com/visva89/pTVreg/tree/master) with DIR_all.m, and run.

After storing the registration results at the appropriate location, sur_2d_np.py is used to extract the surface depth map.

Finally, run the main_surface_{dataset_name}.py corresponding to the dataset for training & test.
