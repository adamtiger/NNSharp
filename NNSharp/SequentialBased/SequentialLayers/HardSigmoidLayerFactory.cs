using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class HardSigmoidLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is HardSigmoid)
                return new HardSigmoidLayer();

            return null;
        }
    }
}
