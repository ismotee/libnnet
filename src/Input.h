#pragma once
#include <memory>

/**
 * @class Input
 * @author Ismo Torvinen
 * @date 02.04.2018
 * @file libnnet.h
 * @brief Input is class for Neuron inputs. Input stores and manages weights and 
 * Input is capable of calculating back propagation from neuron base delta values.
 */
class Neuron;

class Input {
    std::shared_ptr<float> in;
    std::shared_ptr<Neuron> linkedNeuron;
    float weight;
public:
    Input();
    Input(std::shared_ptr<float> in_);
    Input(std::shared_ptr<Neuron> neuron);
    // input control methods
    void connect(Input& in_);
    void connect(std::shared_ptr<Neuron> neuron);
    void connect(std::shared_ptr<float> in_);
    void disconnect();
    // weight control methods
    float getWeight();
    float getWeightedInput();
    void setWeight(float w);
    std::shared_ptr<float> getInput();
    std::shared_ptr<Neuron> getNeuron();
};
