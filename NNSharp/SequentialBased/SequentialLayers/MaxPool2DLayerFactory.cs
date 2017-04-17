using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.KernelDescriptors;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class MaxPool2DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is MaxPooling2D)
            {
                MaxPooling2D conv = descriptor as MaxPooling2D;

                ILayer layer = new MaxPool2DLayer(conv.PaddingVertical, conv.PaddingHorizontal,
                                               conv.StrideVertical, conv.StrideHorizontal, 
                                               conv.KernelHeight, conv.KernelWidth);

                return layer;
            }

            return null;
        }
    }
}
