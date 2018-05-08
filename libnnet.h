#pragma once
#include <vector>
#include <memory>

enum LINK_METHOD {RANDOM, RANDOM_EVERY_NEURON_ONCE , ALL, ALL_ONCE};
enum LAYER_ROLE {INPUT, HIDDEN, OUTPUT};

/**
 * @class Input
 * @author Ismo Torvinen
 * @date 02.04.2018
 * @file libnnet.h
 * @brief Input is class for Neuron inputs. Input sotres and manages weights and 
 * Input is capable of calculating back propagation from neuron base delta values.
 */

class Input
{
	std::shared_ptr<float> in;
	float weight;
public:
	Input ();
	Input (std::shared_ptr<float> in_);
// input control methods
	void connect (Input& in_);
	void connect (std::shared_ptr<float> in_);
	void disconnect ();
// weight control methods
	float getWeight ();
	float getWeightedInput ();
	void setWeight (float w);
	std::shared_ptr<float> getInput ();
};

/**
 * @class Neuron
 * @author Ismo
 * @date 02.04.2018
 * @file libnnet.h
 * @brief Neuron with dynamic amount of inputs and one output. Neuron stores the brief history of Neuron activity.
 * Neuron takes care of inhibition as well as data that is needed for back propagation training of Input.
 */


class Neuron
{
	float learningRate;
	std::vector<Input> inputs;
	float output;
	float bias;
	float error;
	std::shared_ptr<float> outputSignal;

public:
	Neuron ();
	Neuron (std::vector<std::shared_ptr<float> > ins_);
	void addInput (std::shared_ptr<float> out);
	void forward ();
	void back (float desiredOut);
	// getters and setters
	std::shared_ptr<float> getOutputSignal();
	float getOutput();
	float getCurrentError();
};

/**
 * @class nLayer
 * @author Ismo
 * @date 17.04.2018
 * @file libnnet.h
 * @brief Base class for a neural net's layer. Class gives some framework 
 * for managing neurons between upper and lower layer. 
 */
 
class NLayer {
	std::shared_ptr<NLayer> upperLayer;
	std::vector<std::shared_ptr<Neuron> > layer;
	
public:
	NLayer ();
	NLayer (int numOfNeurons);
	void link (std::shared_ptr<NLayer> _upperLayer, void* (*method)(std::shared_ptr<NLayer> _upLayer));  // predicate for linker function 
	virtual void forward() = 0;
	virtual void back(std::vector<float> desiredOutputs) = 0;
};

class InputLayer : public NLayer
{	
public:
	InputLayer();
	void link(std::shared_ptr<float> input);
	void forward();
	void back(std::vector<float> desiredOutputs);
};

/**
 * @class NNet
 * @author Ismo
 * @date 07.05.2018
 * @file libnnet.h
 * @brief General neural net class. NNet has one input, one output layer and
 * dynamic amount of hidden layers.		 
 */

class NNet {
	std::shared_ptr<NLayer> inputLayer;
	std::shared_ptr<NLayer> outputLayer;
	std::vector<std::shared_ptr<NLayer> > hiddenLayers;
	
public:
	NNet();
	NNet(int numOfInputs, int numOfOutputs, int numOfHiddenLayers, int numOfHiddenNeurons);
	void linkInput (std::vector<std::shared_ptr<float> > input);
	void linkHidden (void* (*method)(std::shared_ptr<NLayer> _upperLayer));
	void linkOutput (void* (*method)(std::shared_ptr<NLayer> _upperLayer));

	void forward();
	void back(std::vector<float> desiredOut);
};