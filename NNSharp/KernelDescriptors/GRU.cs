using NNSharp.Kernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.KernelDescriptors
{
    public class GRU : IKernelDescriptor
    {
        public GRU(int units, int inputDim, ActivationLambda activation,
            ActivationLambda recurrentActivation)
        {
            this.units = units;
            this.inputDim = inputDim;
            this.activation = activation;
            this.recurrentActivation = recurrentActivation;
        }

        public int Units { get { return units; } }
        public int InputDim { get { return inputDim; } }
        public ActivationLambda Activation { get { return activation; } }
        public ActivationLambda RecurrentActivation { get { return recurrentActivation; } }

        private int units;
        private int inputDim;
        private ActivationLambda activation;
        private ActivationLambda recurrentActivation;
    }
}
