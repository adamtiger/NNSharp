using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.KernelDescriptors;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class Conv2DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is Convolution2D)
            {
                Convolution2D conv = descriptor as Convolution2D;

                ILayer layer = new Conv2DLayer(conv.PaddingVertical, conv.PaddingHorizontal,
                                               conv.StrideVertical, conv.StrideHorizontal);

                return layer;
            }

            return null;
        }
    }
}
