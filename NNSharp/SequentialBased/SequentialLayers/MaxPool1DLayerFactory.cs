using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class MaxPool1DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is MaxPooling1D)
            {
                MaxPooling1D pool = descriptor as MaxPooling1D;

                ILayer layer = new MaxPool1DLayer(pool.Padding, pool.Stride, pool.KernelSize);

                return layer;
            }

            return null;
        }
    }
}
