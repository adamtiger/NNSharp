using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class SoftsignLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is Softsign)
                return new SoftsignLayer();

            return null;
        }
    }
}
