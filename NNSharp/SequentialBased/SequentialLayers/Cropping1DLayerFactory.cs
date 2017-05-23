using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class Cropping1DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is Cropping1D)
            {
                Cropping1D crop = descriptor as Cropping1D;

                ILayer layer = new Cropping1DLayer(crop.TrimBegin, crop.TrimEnd);

                return layer;
            }

            return null;
        }
    }
}
