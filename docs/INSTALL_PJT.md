# PJT Installation

## FOLDERS 
1) pjt, where the code of PJT is located and resulting library will be generated
2) swigwin-4.0.2, where SWIG is located. SWIG is open-source free library which automatically
 generates a wrapper Python file for Python library (with an extension .pyd), and special .cxx file
 with low-level conversions between types of C++ and Python
 
## SOFTWARE NEEDED 
1)	For compilation one needs to have some Python distributive, which includes the header file Python.h and the library file pythonX.lib, where X is the version of Python. For example, one can use Anaconda distributive:
[https://www.anaconda.com/](https://www.anaconda.com/)  
2)	Also, one should have some C++ compiler. The simplest way is to use open-source g++ compiler. For windows it is included in MinGW distributive:
[https://sourceforge.net/projects/mingw-w64/files/latest/download](https://sourceforge.net/projects/mingw-w64/files/latest/download)

## COMPILATION 
In root folder “PJT_to_Huawei” open file “compile.bat” by any text editor and set up the following variables:  
`ANACONDA_DIR` = “full path to the folder with Anaconda”  
`MINGW_PATH` = “full path to the folder with mingw32-make.exe”  
The default values of these variables are
```
ANACONDA_DIR=C:\Users\peband\AppData\Local\Continuum\anaconda3  
MINGW_PATH=C:\Program Files\mingw64\bin
```

So the following variable will be determined:  
* ```PYTHON_INCLUDE=%ANACONDA_DIR%\include``` - full path to the folder containing header file Python.h
* ```PYTHON_LIB=%ANACONDA_DIR%\libs ``` - full path to the folder containing library file pythonX.lib where X – the version of Python


Open Anaconda Promt (console)
Change directory to the root folder “PJT_to_Huawei” by using console command `cd` 
For example,
```
cd C:\UserName\Documents\PJT_to_Huawei
```

Run in console the command  
```
compile.bat
```

## TEST 
Run in console the command 
```
python test.py
```

## DESCRIPTION OF METHOD 
PJT is callable from the Python package named “pjt”. One need to call the function named “pjt” from this package. 
This function has the following interface:
```
pjt(D, q, T1, T2, M, contSpec, omp_num_threads=1)
```
where   
`D` – the size of the array for the signal  
`q` – complex array for the signal  
`T1`, `T2` – left and right bounds of temporal window  
`M` – the size of continuous spectrum array (if the continuous spectrum was calculated before, 
one can use it to speedup PJT by using data from the real axis, otherwise set `M=0`)  
`contSpec` – complex array with continuous spectrum of size 3*M: (frequencies xi, a(xi), b(xi))  
`omp_num_threads` – the variable to control OpenMP threads number inside PJT  

