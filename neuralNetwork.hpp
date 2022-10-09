#pragma once

#include "Matrix.hpp"
#include <cstdlib>

/*
            1         2      n
    [in] -> [10]    [20]    ...
                    [21]    ...     []  -> out
    [in] -> [11]    [22]    ...

        [10] -> [w10,20], [w10,21], [w10,22] // weight
        [11] -> [w11,20], [w11,21], [w11,22]

        [20] = activate( [10] * [w10,20] + [11] * [w11,20] );
        [21] = activate( [10] * [w10,21] + [11] * [w11,21] );
        [22] = activate( [10] * [w10,22] + [11] * [w11,22] );

        [3 * 2]

        feedForward() -> output;
        error = target - output; (minimization of err) <- minima
        derr = d/dw (error) = + , -
        derr = derr * learningRate;
        transpose(values) [....]
        dw= [.] * derr;
            [.]
            [.]
        weight = weight + dw;
 */

inline double Sigmoid(double x){
    return 1.0f / ( 1 + exp(-x) );
}

inline double DSigmoid(double x){
    return x * ( 1.0f - x );
}


class SimpleNeuralNetwork
{
    public:
        std::vector<u_int32_t> _topology;
        std::vector<Matrix> _weightMatrices;
        std::vector<Matrix> _valueMatrices;
        std::vector<Matrix> _biasMatrices;
        double _learningRate;
    public:
        SimpleNeuralNetwork(std::vector<u_int32_t> topology, double learningRate = 0.1f)
            :_topology(topology),
            _weightMatrices({}),
            _valueMatrices({}),
            _biasMatrices({}),
            _learningRate(learningRate)

        {
            for(u_int32_t i=0; i < topology.size() - 1; i++){
                Matrix weightMatrix(topology[i + 1], topology[i]);
                weightMatrix = weightMatrix.applyFunction(
                    [](const double & val){
                        return (double) rand() / RAND_MAX;
                    }
                );
                _weightMatrices.push_back(weightMatrix);
                Matrix biasMatrix(topology[i + 1], 1);
                biasMatrix = biasMatrix.applyFunction(
                    [](const double & val){
                        return (double) rand() / RAND_MAX;
                    }
                );
                _biasMatrices.push_back(biasMatrix);
            }
            _valueMatrices.resize(topology.size());
        }

        bool feedForward(std::vector<double> input){
            if(input.size() != _topology[0])
                return false;
            
            Matrix values(input.size(), 1);
            for(u_int32_t i = 0; i < input.size(); i++)
                values._vals[i] = input[i];
            
            // Feed forward to next layers
            for(u_int32_t i = 0; i < _weightMatrices.size(); i++){
                _valueMatrices[i] = values;
                values = values.multiply(_weightMatrices[i]);
                values = values.add(_biasMatrices[i]);
                values = values.applyFunction(Sigmoid);
            }

            _valueMatrices[_weightMatrices.size()]  = values;

            return true;
        }

        bool backPropagate(std::vector<double> targetOutput){
            if( targetOutput.size() != _topology.back() )
                return false;
            Matrix errors(targetOutput.size(), 1);
            errors._vals = targetOutput;
            Matrix sub = _valueMatrices.back().negative();
            errors = errors.add(sub);
            for(int32_t i = _weightMatrices.size() - 1; i >= 0; i --){
                Matrix trans = _weightMatrices[i].transpose();
                Matrix prevErrors = errors.multiply( trans );

                Matrix dOutputs = _valueMatrices[i + 1].applyFunction( DSigmoid );
                Matrix gradients = errors.multiplyElements(dOutputs);
                gradients = gradients.multiplyScaler( _learningRate );
                Matrix weightGradients = _valueMatrices[i].transpose().multiply(gradients);
                
                _biasMatrices[i] = _biasMatrices[i].add(gradients);
                _weightMatrices[i] = _weightMatrices[i].add(weightGradients);
                
                errors = prevErrors;
            }
            return true;
        }

        std::vector<double> getPrediction(){
            return _valueMatrices.back()._vals;
        }
};