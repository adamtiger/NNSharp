using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class AvgPool2DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is AvgPooling2D)
            {
                AvgPooling2D conv = descriptor as AvgPooling2D;

                ILayer layer = new AvgPool2DLayer(conv.PaddingVertical, conv.PaddingHorizontal,
                                               conv.StrideVertical, conv.StrideHorizontal,
                                               conv.KernelHeight, conv.KernelWidth);

                return layer;
            }

            return null;
        }
    }
}
