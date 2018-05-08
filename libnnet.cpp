#include "libnnet.h"
#include <random>
#include <algorithm>
#include <math.h>
#include <iostream>

/**********************************************************************/
/*** sigmoid.c:  This code contains the function routine            ***/
/***             sigmoid() which performs the unipolar sigmoid      ***/
/***             function for backpropagation neural computation.   ***/
/***             Accepts the input value x then returns it's        ***/
/***             sigmoid value in float.                            ***/
/***                                                                ***/
/***  function usage:                                               ***/
/***       float sigmoid(float x);                                  ***/
/***           x:  Input value                                      ***/
/***                                                                ***/
/***  Written by:  Kiyoshi Kawaguchi                                ***/
/***               Electrical and Computer Engineering              ***/
/***               University of Texas at El Paso                   ***/
/***  Last update:  09/28/99  for version 2.0 of BP-XOR program     ***/
/**********************************************************************/


float sigmoid (float x)
{
     float exp_value;
     float return_value;

     /*** Exponential calculation ***/
     exp_value = exp((double) -x);

     /*** Final sigmoid value ***/
     return_value = 1 / (1 + exp_value);

     return return_value;
}



Input::Input ()
{
	weight = rand() % 100000 * 0.00001;
	std::cout << "weight: " << weight << "\n";
}

Input::Input (std::shared_ptr<float> in_)
{
	weight = rand() % 100000 * 0.00001;
	std::cout << "weight: " << weight << "\n";
	in = in_;
}

void Input::connect (std::shared_ptr<float> in_)
{
	in = in_;
}

void Input::disconnect ()
{
	in = nullptr;
}

float Input::getWeight ()
{
	return weight;
}

float Input::getWeightedInput ()
{
	return weight * (*in);
}

void Input::setWeight (float w)
{
	weight = w;
}

std::shared_ptr<float> Input::getInput() {
	return in;
}

Neuron::Neuron (): learningRate(0.01), bias(1), error(0)
{
	outputSignal = std::make_shared<float>(0);
}

Neuron::Neuron (std::vector<std::shared_ptr<float> > outs)
{
	for(auto& out : outs)
	{
		addInput(out);
	}
}

void Neuron::addInput (std::shared_ptr<float> out)
{
	inputs.push_back(Input(out));
}

std::shared_ptr<float> Neuron::getOutputSignal()
{
	return outputSignal;
}

float Neuron::getOutput() {
	return output;
}

void Neuron::forward ()
{
	output = bias;
	for(auto& input : inputs)
	{
		output += input.getWeightedInput();
	}
	(*outputSignal) = sigmoid(output);
}

void Neuron::back (float desiredOut)
{
	error = desiredOut - (*outputSignal);
	
	for(auto& input : inputs)
	{
		// sigmoidDelta is sig(in) - (1 - sig(in)) => last input == sig(in)
		float sigmoidDelta = (*input.getInput()) - (1 - (*input.getInput()));
		float weightDelta = -error * sigmoidDelta * (*input.getInput());
		input.setWeight(input.getWeight() + weightDelta * learningRate);
	}
}

float Neuron::getCurrentError()
{
	return error;
}

NLayer::NLayer() 
{
}

NLayer::NLayer(int numOfNeurons) 
{
	for(int i = 0; i < numOfNeurons; i++) 
	{
		layer.push_back(std::make_shared<Neuron>());
	}
}

void NLayer::link(std::shared_ptr<NLayer> _upperLayer, void* (*method)(std::shared_ptr<NLayer> _upLayer)) 
{
	upperLayer = _upperLayer;
	method(_upperLayer);
}

InputLayer::forward()
{
	for(auto& neuron : layer)
	{
		neuron.forward();
	}
}

InputLayer::back(std::vector<float> desiredOuts)
{
	if(desiredOuts.size() != layer.size()) {
		std::cerr << "Input Layer: mismatch of vector size: " << layer.size() << "/" << desiredOuts.size() << "\n"; 
	}
	for(int i = 0; i < layer.size(); i++)
	{
		layer[i].back(desiredOuts[i]);
	}
}


NNet::NNet() 
{
	
}

NNet::NNet(int numOfInputs, int numOfOutputs, int numOfHiddenLayers, int numOfHiddenNeurons)
{
	inputLayer = std::make_shared<NLayer>(numOfInputs);
	outputLayer = std::make_shared<NLayer>(numOfOutputs);
	hiddenLayers.push_back(std::make_shared<NLayer>(numOfHiddenNeurons));
	
	for(int i = 1; i < numOfHiddenLayers; i++) {
		hiddenLayers.push_back(std::make_shared<NLayer>(numOfHiddenNeurons));
	}
	
	outputLayer = std::make_shared<NLayer>(numOfOutputs);
	
}

void NNet::linkHidden(void* (*method)(std::shared_ptr<NLayer> _upLayer))
{
}

void NNet::linkOutput(void* (*method)(std::shared_ptr<NLayer> _upLayer))
{
}

void NNet::linkInput(std::vector<std::shared_ptr<float> > input)
{
	
}

void NNet::forward()
{
}

void NNet::back(std::vector<float> desiredOut)
{
}
