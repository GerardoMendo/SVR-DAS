# SVR-DAS
ML-based method for earthquake detection and seismic phase picking from Distributed Acoustic Sensing data. 

Results obtained using this code are published in the following reference: 

Mendo - Perez, G., Nagao, H., Katoh, S., Shinohara, M. (2026). SVR-DAS: A Machine-Learning-Based Method to Create Earthquake Catalogs from Seafloor Distributed Acoustic Sensing measurements. Submitted to Earth, Planets and Space. 

When using this code, please cite this reference. 

# Requirements
1. Python >= 3.10
2. NumPy
3. SciPy
4. SciKit-Learn
5. h5py

# How to run

1. Modify the dict_params.py file according to the required parameters for both detection and picking.
2. Set the running mode (detection only, picking only, both). Inside the main.py file you can set the download variable to True
   or False, depending on the mode you want. If both True, the program will first run the detection module, and once it finished
   to search in all data it will proceed to execute the picking module.
3. Run main.py

# Considerations

1. It is possible to run data locally or by downloading data from server. If download from a server, configure download_data.py
   accordingly to the server access credentials.
2. If downloading the data, the execution time increases drastically.
3. When downloading data, uncomment the path_wav file in dict_params.py that contains the file 'temp' and comment or erase the other ones.
4. If downloading the data, the program will erase the downloaded data. This ensures that the disk memory will keep the same.
5. This program was made considering DAS data structure from OptaSense interrogators. Current modifications are being done to this program to allow     to use other data formats.
 
