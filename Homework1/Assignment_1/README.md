## Build and execute
In the source folder: 
 
 1. `mkdir build`, **only do once**. 
 1. `cd build; cmake ..` to create the Makefile, **only do once**
    1. If cmake failed to locate where OpenCV is installed, you can set it 
       by setting the OpenCV_DIR env: `export OpenCV_DIR=/opt/opencv`
 1. `make` to build the examples. **Do each time you change your source code, and want to build and execute the program** 
 1. execute an example, e.g. `./Histogram`

