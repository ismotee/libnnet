#include "libnnet.h"
#include <iostream>
#include <memory>
#include <cmath>

int lineLength = 100;

int main(int argc, char **argv)
{
	std::vector<float> line;
	for (int i = 0; i < lineLength; i++) 
	{
		line.push_back(i % 3);
		if(line.back() != 1) 
		{
			line.back() = 0;
		}
	}

	std::cout << "making input signals\n";
	std::shared_ptr<float> in1,in2;
	in1 = std::make_shared<float>(1);
	
	std::cout << "creating neurons\n";
	Neuron f,s,t;
	
	std::cout << "linking neurons\n";
	f.addInput(in1);
	s.addInput(f.getOutputSignal());
	f.addInput(s.getOutputSignal());
	t.addInput(s.getOutputSignal());
	s.addInput(t.getOutputSignal());

	for( int i = 0; i < line.size(); i++) {
		std::cout << "sending forward call\n";

		*in1 = line[i];

		t.forward();
		s.forward();
		f.forward();
				
		std::cout << "neuron output: " << (*t.getOutputSignal()) << " & " << "\n";
		if(i + 3 > lineLength)
		{
			f.back(line[i+1]);
			s.back(line[i+2]);
			t.back(line[i+3]);
		}
		
	}
	return 0;
}
