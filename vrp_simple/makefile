build:main.o input.o read_data_from_txt.o VRP.o
	g++ -o result.out main.o input.o read_data_from_txt.o VRP.o
main.o:main.cpp input.hpp
	g++ -g -c -lstdc++ main.cpp
input.o:input.cpp read_data_from_txt.hpp
	g++ -g -c -lstdc++ input.cpp
read_data_from_txt.o:read_data_from_txt.cpp
	g++ -g -c -lstdc++ read_data_from_txt.cpp
VRP.o:VRP.cpp
	g++ -g -c -lstdc++ VRP.cpp
clean:
	rm main.o input.o read_data_from_txt.o VRP.o