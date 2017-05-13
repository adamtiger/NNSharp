using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class AvgPool1DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is AvgPooling1D)
            {
                AvgPooling1D pool = descriptor as AvgPooling1D;

                ILayer layer = new AvgPool1DLayer(pool.Padding, pool.Stride, pool.KernelSize);

                return layer;
            }

            return null;
        }
    }
}
