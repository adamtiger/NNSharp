using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class MinPool2DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is MinPooling2D)
            {
                MinPooling2D conv = descriptor as MinPooling2D;

                ILayer layer = new MaxPool2DLayer(conv.PaddingVertical, conv.PaddingHorizontal,
                                               conv.StrideVertical, conv.StrideHorizontal,
                                               conv.KernelHeight, conv.KernelWidth);

                return layer;
            }

            return null;
        }
    }
}
