swig2.0 -Wall -c++ -python pcSavePython.i

cd build
cmake ..
make

cp libsave_pc.so ../_pcSavePython.so

cd ..

cp _pcSavePython.so pcSavePython.py ~/projects/ros/baris/src/
