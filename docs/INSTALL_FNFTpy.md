 * you may install FNFTpy locally using **pip**:
   From within the project root folder run
     ```
     pip install .     # Install system wide
     pip install -e .  # Install in editable/development mode
     ```

 * **alternatively**, you may add the FNFTpy folder to your Python path.

 * Of course, you need a compiled version of the FNFT C-library. See the
  [documentation for FNFT](https://github.com/FastNFT/FNFT) on how to build 
   the library on your device. 
 * FNFTpy needs to know where the C-library is located. 
   This configuration can be done by editing the function get_lib_path()
   in auxiliary.py. 
   
   Example:
    ```   
    def get_lib_path():
        """Return the path of the FNFT file.
    
        Here you can set the location of the compiled library for FNFT.
        See example strings below.
    
        Returns:
    
        * libstring : string holding library path
    
        Example paths:
    
            * libstr = "C:/Libraries/local/libfnft.dll"  # example for windows            
            * libstr = "/usr/local/lib/libfnft.so"  # example for linux
    
        """
        libstr = "/usr/local/lib/libfnft.so"  # example for linux
        return libstr        
    ```
    
