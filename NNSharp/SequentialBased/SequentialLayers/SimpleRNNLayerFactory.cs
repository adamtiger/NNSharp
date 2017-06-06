using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class SimpleRNNLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is SimpleRNN)
            {
                SimpleRNN rnn = descriptor as SimpleRNN;

                ILayer layer = new SimpleRNNLayer(rnn.Units, rnn.InputDim, rnn.Lambda);

                return layer;
            }

            return null;
        }
    }
}
