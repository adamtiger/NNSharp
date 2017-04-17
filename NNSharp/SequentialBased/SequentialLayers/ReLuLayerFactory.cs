using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.KernelDescriptors;
using NNSharp.DataTypes;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class ReLuLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is ReLu)
                return new ReLuLayer();

            return null;
        }
    }
}
