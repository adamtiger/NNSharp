using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class Conv1DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is Convolution1D)
            {
                Convolution1D conv = descriptor as Convolution1D;

                ILayer layer = new Conv1DLayer(conv.Padding, conv.Stride);

                return layer;
            }

            return null;
        }
    }
}
