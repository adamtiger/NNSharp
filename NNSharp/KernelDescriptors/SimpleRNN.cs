using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.KernelDescriptors
{
    public delegate void ActivationLambda(IData data);

    public class SimpleRNN : IKernelDescriptor
    {
        public SimpleRNN(int units, int inputDim, ActivationLambda lambda)
        {
            this.units = units;
            this.inputDim = inputDim;
            this.lambda = lambda;
        }

        public int Units { get { return units; } }
        public int InputDim { get { return inputDim; } }
        public ActivationLambda Lambda { get { return lambda; } }

        private int units;
        private int inputDim;
        private ActivationLambda lambda;
    }
}
