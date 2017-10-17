using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.KernelDescriptors;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class DropoutLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is Dropout)
            {
                Dropout dropout = descriptor as Dropout;

                ILayer layer = new DropoutLayer(dropout.Rate, dropout.NoiseShape);

                return layer;
            }

            return null;
        }
    }
}
