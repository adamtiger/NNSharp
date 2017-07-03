using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class LSTMLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is LSTM)
            {
                LSTM rnn = descriptor as LSTM;

                ILayer layer = new LSTMLayer(rnn.Units, rnn.InputDim, rnn.Activation,
                                           rnn.RecurrentActivation);

                return layer;
            }

            return null;
        }
    }
}
