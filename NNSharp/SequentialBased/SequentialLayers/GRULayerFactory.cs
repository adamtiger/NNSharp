using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class GRULayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is GRU)
            {
                GRU rnn = descriptor as GRU;

                ILayer layer = new GRULayer(rnn.Units, rnn.InputDim, rnn.Activation,
                                           rnn.RecurrentActivation);

                return layer;
            }

            return null;
        }
    }
}
