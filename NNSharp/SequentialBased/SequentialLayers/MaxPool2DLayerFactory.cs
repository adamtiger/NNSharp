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
                MaxPooling2D pool = descriptor as MaxPooling2D;

                ILayer layer = new MaxPool2DLayer(pool.PaddingVertical, pool.PaddingHorizontal,
                                               pool.StrideVertical, pool.StrideHorizontal, 
                                               pool.KernelHeight, pool.KernelWidth);

                return layer;
            }

            return null;
        }
    }
}
