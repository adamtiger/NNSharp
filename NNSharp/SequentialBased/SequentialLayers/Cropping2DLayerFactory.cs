using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class Cropping2DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is Cropping2D)
            {
                Cropping2D crop = descriptor as Cropping2D;

                ILayer layer = new Cropping2DLayer(crop.TopTrim, crop.BottomTrim, crop.LeftTrim, crop.RightTrim);

                return layer;
            }

            return null;
        }
    }
}
